import os
import json
import random
import torch
import gc
import signal
import argparse
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from tqdm import tqdm

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutException("Generation timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def clear_gpu_memory():
    """Clear GPU memory to prevent out-of-memory errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def extract_assistant_response(text):
    """Extract the assistant's response from the full text."""
    # Try different formats based on Qwen's possible response formats
    if "<|im_start|>assistant" in text:
        parts = text.split("<|im_start|>assistant")
        if len(parts) > 1:
            assistant_part = parts[1]
            if "<|im_end|>" in assistant_part:
                return assistant_part.split("<|im_end|>")[0].strip()
            else:
                return assistant_part.strip()
    elif "<|assistant|>" in text:
        return text.split("<|assistant|>")[-1].strip()
    elif "ASSISTANT:" in text:
        return text.split("ASSISTANT:")[-1].strip()
    
    # If no specific markers found, return the whole text
    return text

def log_with_timestamp(message):
    """Log a message with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_response(model, tokenizer, system_message, user_prompt, args):
    """Generate a response from the model."""
    generation_start_time = time.time()
    
    try:
        with time_limit(args.timeout):
            try:
                log_with_timestamp(f"Starting generation for prompt: '{user_prompt[:50]}...'")
                
                # First try using the built-in chat method if available
                if hasattr(model, 'chat') and callable(getattr(model, 'chat')):
                    log_with_timestamp("Using model's built-in chat method")
                    # Qwen2.5 chat-specific method
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = model.chat(tokenizer, messages, temperature=args.temperature, max_new_tokens=args.max_length)
                    model_response = response
                    log_with_timestamp(f"Response generated using built-in chat method (length: {len(model_response)})")
                else:
                    log_with_timestamp("Using manual chat template and generation")
                    # Fallback to manual chat template and generation
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Apply chat template
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    log_with_timestamp(f"Template applied. Input length: {len(prompt)} chars")
                    
                    # Tokenize
                    tokenize_start = time.time()
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    log_with_timestamp(f"Tokenization completed in {time.time() - tokenize_start:.2f}s. Input tokens: {inputs.input_ids.shape[1]}")
                    
                    # Generate
                    generate_start = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=args.max_length,
                            temperature=args.temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            repetition_penalty=1.1  # Add slight penalty to avoid repetitions
                        )
                    log_with_timestamp(f"Generation completed in {time.time() - generate_start:.2f}s. Output tokens: {outputs.shape[1]}")
                    
                    # Decode
                    decode_start = time.time()
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    log_with_timestamp(f"Decoding completed in {time.time() - decode_start:.2f}s")
                    
                    # Extract assistant response
                    extract_start = time.time()
                    model_response = extract_assistant_response(full_response)
                    log_with_timestamp(f"Response extraction completed in {time.time() - extract_start:.2f}s")
                    
                    # If extraction failed, use the full response
                    if not model_response:
                        log_with_timestamp("Extraction failed, using full response")
                        model_response = full_response
            
            except Exception as e:
                log_with_timestamp(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                model_response = f"Error: {str(e)}"
    
    except TimeoutException as e:
        log_with_timestamp(f"Generation timed out after {args.timeout} seconds.")
        model_response = f"Error: Generation timed out after {args.timeout} seconds."
    
    generation_time = time.time() - generation_start_time
    log_with_timestamp(f"Total generation time: {generation_time:.2f}s. Response length: {len(model_response)} chars")
    
    return model_response, generation_time

def save_conversation(conversation, output_dir, count):
    """Save the conversation to files."""
    save_start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON output
    json_filename = os.path.join(output_dir, f"conversation_{count:04d}_{timestamp}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    # TXT output
    txt_filename = os.path.join(output_dir, f"conversation_{count:04d}_{timestamp}.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"Conversation #{count}\n")
        f.write(f"Timestamp: {conversation['timestamp']}\n\n")
        f.write(f"System Message:\n{conversation['system_message']}\n\n")
        f.write(f"User Prompt:\n{conversation['user_prompt']}\n\n")
        f.write(f"Model Response:\n{conversation['model_response']}\n\n")
        if 'generation_time' in conversation:
            f.write(f"Generation Time: {conversation['generation_time']:.2f} seconds\n")
    
    log_with_timestamp(f"Conversation saved to {json_filename} and {txt_filename} in {time.time() - save_start_time:.2f}s")
    return json_filename, txt_filename

def main(args):
    # Start timing
    total_start_time = time.time()
    
    # Model path
    model_path = args.model_path
    
    # Check if model files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load model and tokenizer
    log_with_timestamp(f"Loading model and tokenizer from {model_path}...")
    load_start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log_with_timestamp(f"Tokenizer loaded in {time.time() - load_start_time:.2f}s")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log_with_timestamp("Set padding token to EOS token")
    
    # Load model with optimizations
    model_load_start = time.time()
    log_with_timestamp("Loading model (this may take several minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Use lower precision to save memory
    )
    model.eval()  # Set model to evaluation mode
    log_with_timestamp(f"Model loaded in {time.time() - model_load_start:.2f}s")
    
    # Report model info
    log_with_timestamp(f"Model device map: {model.hf_device_map}")
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    log_with_timestamp(f"GPU memory usage: {gpu_mem:.2f} GB")
    
    # System message
    system_message = """**请基于真实世界信息，生成一个在上海市陆家嘴区域内进行活动的人，在某一典型工作日内的完整活动轨迹信息。要求：时间安排符合上海都市生活作息规律，空间位置限定在上海市陆家嘴区域内，活动轨迹需体现通勤、工作、餐饮、休闲等日常行为，且符合现代都市生活的真实场景与逻辑。坐标信息均为1984坐标系。**"""
    
    # List of input prompts
    input_prompts = [
        "请生成一份陆家嘴区域内某人的日常活动轨迹信息。",
        "生成一个在陆家嘴活动的人的一天行程记录。",
        "描述一位在陆家嘴区域内活动者的全天行程轨迹。",
        "请给出陆家嘴区域内某人一天的活动轨迹信息。",
        "生成一份记录陆家嘴区域内某人全天活动的轨迹信息。",
        "请生成陆家嘴内一位人士的一天活动轨迹。",
        "生成陆家嘴区域内某人的一日活动轨迹记录。",
        "描述陆家嘴区域内一位活动者的全天行程。",
        "请给出陆家嘴区域内某人的全天活动轨迹。",
        "生成陆家嘴内某人一天的行程信息。",
        "请生成陆家嘴区域内某人的日常行程记录。",
        "描述一个在陆家嘴活动的人的全天轨迹。",
        "生成陆家嘴区域内一位人士全天的活动记录。",
        "请提供陆家嘴区域内某人一天的详细行程。",
        "生成一份关于陆家嘴内某人全天活动的轨迹记录。",
        "请描述陆家嘴区域内一位人士的全天行程。",
        "生成陆家嘴区域内某人的日常活动轨迹。",
        "请生成一份记录陆家嘴内某人全天行程的提示信息。",
        "生成陆家嘴内某人一日活动的轨迹记录。",
        "请给出陆家嘴区域内某人一天行程的完整轨迹信息。",
        "Please generate a detailed daily itinerary for a person active in the Lujiazui area.",
        "Generate an activity log for a person spending their day in Lujiazui, Shanghai.",
        "Describe the daily journey of someone who works and lives in Lujiazui.",
        "Provide a full-day activity record for an individual in the Lujiazui district.",
        "Produce a daily schedule outlining various activities of a person in Lujiazui."
    ]
    
    # Create summary file
    summary_file = os.path.join(args.output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Inference Summary\n")
        f.write(f"================\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Max Sequence Length: {args.max_length}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Timeout: {args.timeout}s\n")
        f.write(f"Target Count: {args.count}\n\n")
    
    log_with_timestamp(f"Created summary file: {summary_file}")
    
    # Main loop for generating responses
    count = 0
    successful_count = 0
    error_count = 0
    total_generation_time = 0
    
    log_with_timestamp(f"Starting generation of {args.count} responses...")
    
    progress_bar = tqdm(total=args.count, desc="Generating responses")
    
    while count < args.count:
        iteration_start = time.time()
        count += 1
        
        # Randomly select a prompt
        selected_prompt = random.choice(input_prompts)
        log_with_timestamp(f"[{count}/{args.count}] Selected prompt: {selected_prompt}")
        
        # Generate response
        try:
            model_response, generation_time = generate_response(model, tokenizer, system_message, selected_prompt, args)
            total_generation_time += generation_time
            
            if model_response.startswith("Error:"):
                error_count += 1
                log_with_timestamp(f"Generation error: {model_response}")
            else:
                successful_count += 1
            
            # Create conversation record
            conversation = {
                "id": count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_message": system_message,
                "user_prompt": selected_prompt,
                "model_response": model_response,
                "generation_time": generation_time
            }
            
            # Save to files
            json_file, txt_file = save_conversation(conversation, args.output_dir, count)
            
            # Update progress bar
            progress_bar.update(1)
            iteration_time = time.time() - iteration_start
            progress_bar.set_postfix({
                "success": successful_count, 
                "errors": error_count, 
                "avg_time": f"{total_generation_time/count:.2f}s",
                "iter_time": f"{iteration_time:.2f}s"
            })
            
            # Add to summary
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"[{count}/{args.count}] Generated response (Time: {generation_time:.2f}s): {txt_file}\n")
            
            # Clear memory after each generation
            clear_gpu_memory()
            
        except Exception as e:
            error_count += 1
            log_with_timestamp(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"success": successful_count, "errors": error_count})
            
            # Add to summary
            with open(summary_file, "a", encoding="utf-8") as f:
                f.write(f"[{count}/{args.count}] ERROR: {str(e)}\n")
    
    progress_bar.close()
    
    # Final statistics
    total_time = time.time() - total_start_time
    average_generation_time = total_generation_time / count if count > 0 else 0
    
    log_with_timestamp(f"Completed {count} conversations")
    log_with_timestamp(f"Successful: {successful_count}, Errors: {error_count}")
    log_with_timestamp(f"Total time: {total_time:.2f}s")
    log_with_timestamp(f"Average generation time: {average_generation_time:.2f}s")
    
    # Update summary with final statistics
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\nFinal Statistics\n")
        f.write(f"===============\n")
        f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {total_time:.2f}s\n")
        f.write(f"Total Conversations: {count}\n")
        f.write(f"Successful Generations: {successful_count}\n")
        f.write(f"Errors: {error_count}\n")
        f.write(f"Average Generation Time: {average_generation_time:.2f}s\n")
    
    log_with_timestamp(f"Summary saved to {summary_file}")
    
    return {
        "total_count": count,
        "successful_count": successful_count,
        "error_count": error_count,
        "total_time": total_time,
        "average_generation_time": average_generation_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Qwen2.5 model")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/ljz_qwen2.5-7b",
                        help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--max_length", type=int, default=5400,
                        help="Maximum sequence length for generation")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation sampling")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds for generation")
    parser.add_argument("--count", type=int, default=600,
                        help="Number of conversations to generate (max 600)")
    
    args = parser.parse_args()
    
    # Ensure count doesn't exceed 600
    args.count = min(args.count, 600)
    
    main(args)
