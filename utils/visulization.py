import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

def parse_log_file(filepath: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Parses the training log file to extract steps, losses, and MSE losses.
    
    Args:
        filepath: Path to the log file (e.g., results/exp_name/log.txt)
    """
    steps = []
    losses = []
    mse_losses = []

    if not os.path.exists(filepath):
        print(f"Log file not found: {filepath}")
        return steps, losses, mse_losses
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"Global Step: (\d+)\s+Loss: ([0-9.]+)\s+MSE_Loss: ([0-9.]+)"
    matches = re.findall(pattern, content)
    
    for match in matches:
        steps.append(int(match[0]))
        losses.append(float(match[1]))
        mse_losses.append(float(match[2]))

    return steps, losses, mse_losses

def plot_training_curves(log_file: str):
    """
    Plots training Loss and MSE Loss curves.
    """
    steps, losses, mse_losses = parse_log_file(log_file)

    if steps:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, label='Total Loss')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(steps, mse_losses, label='MSE Loss', color='orange')
        plt.xlabel('Global Step')
        plt.ylabel('MSE')
        plt.title('MSE Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("No log data found to plot.")

def parse_eval_results(filepath: str) -> pd.DataFrame:
    """
    Parses the evaluation_results.txt file into a DataFrame.
    """
    data = []
    if not os.path.exists(filepath):
        # print(f"Evaluation results file not found: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_step = None
    current_nll = None
    current_bpd = None
    current_num_samples = None

    for line in lines:
        line = line.strip()
        if line.startswith("Step"):
            try:
                current_step = int(line.split()[1].replace(':', ''))
            except (IndexError, ValueError):
                continue
        elif line.startswith("Num Samples"):
            try:
                current_num_samples = int(line.split(':')[1])
            except (IndexError, ValueError):
                current_num_samples = None
        elif line.startswith("NLL"):
            try:
                current_nll = float(line.split(':')[1])
            except (IndexError, ValueError):
                pass
        elif line.startswith("BPD"):
            try:
                current_bpd = float(line.split(':')[1])
            except (IndexError, ValueError):
                pass
            
            if current_step is not None:
                entry = {
                    "Step": current_step, 
                    "NLL": current_nll, 
                    "BPD": current_bpd
                }
                if current_num_samples is not None:
                    entry["Num Samples"] = current_num_samples
                
                data.append(entry)
                current_step = None
                current_num_samples = None

    return pd.DataFrame(data)

def show_eval_results(eval_results_file: str, exp_name: str, ckpt_step: Optional[int] = None, show_all: bool = False, batch_size: int = 100, limit_batches: Optional[int] = None):
    """
    Displays evaluation results table and plots BPD curve. Automatically runs evaluate.py if necessary.
    
    Args:
        eval_results_file: Path to the results file.
        exp_name: Experiment name.
        ckpt_step: Specific checkpoint step to view/evaluate. If None and show_all is False, defaults to the latest.
        show_all: If True, shows/evaluates all checkpoints.
        batch_size: Batch size for evaluation.
        limit_batches: Limit number of batches to evaluate (for fast testing).
    """

    # 1. Check existing results
    df_eval = parse_eval_results(eval_results_file)

    # 2. Decide whether to run the evaluation script
    needs_eval = False
    # Assume evaluate.py is in the current working directory or root
    eval_script = "evaluate.py"
    if not os.path.exists(eval_script):
        # Try finding it in the parent directory (if called from utils/)
        if os.path.exists(os.path.join("..", eval_script)):
            eval_script = os.path.join("..", eval_script)

    eval_args = ["python", eval_script, "--exp_name", exp_name]

    if batch_size is not None:
        eval_args.extend(["--batch_size", str(batch_size)])
    if limit_batches is not None:
        eval_args.extend(["--limit_batches", str(limit_batches)])

    if show_all:
        # If file is empty or missing, must run
        if df_eval.empty:
            needs_eval = True
            eval_args.append("--ckpt_all")
    else:
        # Target specific Step or Latest Step
        target_step = ckpt_step
        
        if target_step is None:
            # Automatically find the latest checkpoint file
            exp_dir = os.path.dirname(eval_results_file)
            ckpt_dir = os.path.join(exp_dir, "ckpt") # Note: "ckpt" in trainer.py, might differ in evaluate.py
            if not os.path.exists(ckpt_dir):
                # Fallback for old directory name
                ckpt_dir = os.path.join(exp_dir, "checkpoints")

            if os.path.exists(ckpt_dir):
                files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and f.startswith("step_")]
                if files:
                    # Format: step_1000.pt
                    target_step = max([int(f.split("_")[1].split(".")[0]) for f in files])

        if target_step is not None:
            # Check if this step is already in the dataframe
            if df_eval.empty or target_step not in df_eval["Step"].values:
                needs_eval = True
                eval_args.extend(["--ckpt_step", str(target_step)])
        else:
            print("No checkpoints found to evaluate.")
            if df_eval.empty:
                return

    if needs_eval:
        print(f"Running evaluation for experiment '{exp_name}'...")
        print(f"Command: {' '.join(eval_args)}")
        try:
            subprocess.run(eval_args, check=True)
            print("Evaluation complete.")
            # Reload results
            df_eval = parse_eval_results(eval_results_file)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            return
        except FileNotFoundError:
            print(f"Could not find python or evaluate.py script.")
            return

    if not df_eval.empty:
        # Sort by Step
        df_eval = df_eval.sort_values("Step")
        
        # Filter display
        if not show_all:
            if ckpt_step is not None:
                df_display = df_eval[df_eval["Step"] == ckpt_step]
            else:
                # Show latest
                df_display = df_eval.iloc[[-1]]
        else:
            df_display = df_eval

        print("Evaluation Results:")
        display(df_display)
        
        # Plot only if multiple data points exist and showing all
        if len(df_eval) > 1 and show_all:
            plt.figure(figsize=(8, 5))
            plt.plot(df_eval["Step"], df_eval["BPD"], marker='o', linestyle='-', color='green')
            plt.xlabel("Checkpoint Step")
            plt.ylabel("Bits Per Dimension (BPD)")
            plt.title("BPD over Training")
            plt.grid(True, alpha=0.3)
            plt.show()
    else:
        print("No evaluation results found.")

def show_generated_samples(images_dir: str):
    """
    Concatenates generated sample images for display (Side-by-side comparison of 1-step and 5-step generation).
    Supported filename formats:
      - Old: 1-step_500.png
      - New: 1-step_500-iter.png
    """
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if not files:
        print("No image samples found.")
        return

    # images_by_step[step] = {'1-step': path, '5-step': path}
    images_by_step = {}

    for f in files:
        step = -1
        type_label = None
        
        # Parse filename
        # Assumed format: "{type_label}_step_{step}[-iter].png"
        # Example: "1-step_1000-iter.png" -> type="1-step", step=1000

        parts = f.split('_')
        # parts[0] could be "1-step" or "5-step"
        if len(parts) >= 2:
            if parts[0] == "1-step":
                type_label = "1-step"
            elif parts[0] == "5-step":
                type_label = "5-step"

            if type_label:
                # Parse step
                # parts[1] could be "1000.png" or "1000-iter.png"
                step_str = parts[1]
                # Remove extension
                if step_str.endswith(".png"):
                    step_str = step_str[:-4]
                # Remove "-iter" suffix (compatible with new trainer)
                if step_str.endswith("-iter"):
                    step_str = step_str[:-5]

                try:
                    step = int(step_str)
                except ValueError:
                    continue

        if step != -1 and type_label:
            if step not in images_by_step:
                images_by_step[step] = {}
            images_by_step[step][type_label] = os.path.join(images_dir, f)

    if not images_by_step:
        print("No 1-step or 5-step images found (checked patterns '1-step_...' and '5-step_...').")
        return

    sorted_steps = sorted(images_by_step.keys())
    
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    label_width = 100
    rows = []
    max_row_width = 0

    for step in sorted_steps:
        path_1 = images_by_step[step].get('1-step')
        path_5 = images_by_step[step].get('5-step')

        img_1 = None
        img_5 = None

        if path_1:
            try:
                img_1 = Image.open(path_1)
            except Exception as e:
                print(f"Error loading {path_1}: {e}")

        if path_5:
            try:
                img_5 = Image.open(path_5)
            except Exception as e:
                print(f"Error loading {path_5}: {e}")

        if img_1 is None and img_5 is None:
            continue

        # Determine dimensions
        h1 = img_1.height if img_1 else 0
        w1 = img_1.width if img_1 else 0
        h5 = img_5.height if img_5 else 0
        w5 = img_5.width if img_5 else 0

        row_height = max(h1, h5)

        # Align width
        placeholder_w = w1 if w1 > 0 else (w5 if w5 > 0 else 0)

        current_w1 = w1 if img_1 else placeholder_w
        current_w5 = w5 if img_5 else placeholder_w

        row_width = label_width + current_w1 + label_width + current_w5
        row_img = Image.new('RGB', (row_width, row_height), color='white')
        draw = ImageDraw.Draw(row_img)

        # --- 1-step section ---
        x_offset = 0
        # Label 1
        text_1 = f"Step: {step}\nType: 1-step"
        bbox1 = draw.multiline_textbbox((0, 0), text_1, font=font)
        text_h1 = bbox1[3] - bbox1[1]
        y_text1 = (row_height - text_h1) // 2
        draw.multiline_text((x_offset + 5, y_text1), text_1, fill='black', font=font)

        x_offset += label_width
        # Image 1
        if img_1:
            y_img1 = (row_height - img_1.height) // 2
            row_img.paste(img_1, (x_offset, y_img1))

        x_offset += current_w1

        # --- 5-step section ---
        # Label 5
        text_5 = f"Step: {step}\nType: 5-step"
        bbox5 = draw.multiline_textbbox((0, 0), text_5, font=font)
        text_h5 = bbox5[3] - bbox5[1]
        y_text5 = (row_height - text_h5) // 2
        draw.multiline_text((x_offset + 5, y_text5), text_5, fill='black', font=font)

        x_offset += label_width
        # Image 5
        if img_5:
            y_img5 = (row_height - img_5.height) // 2
            row_img.paste(img_5, (x_offset, y_img5))

        rows.append(row_img)
        max_row_width = max(max_row_width, row_width)

    if not rows:
        return

    # Concatenate rows
    total_height = sum(r.height for r in rows)
    final_image = Image.new('RGB', (max_row_width, total_height), color='white')

    y_curr = 0
    for r in rows:
        final_image.paste(r, (0, y_curr))
        y_curr += r.height

    # Display
    plt.figure(figsize=(20, len(rows) * 4)) # Wider figure for side-by-side
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()