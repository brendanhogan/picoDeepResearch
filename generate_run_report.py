import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

# --- Plot Styling ---
def apply_retro_futurism_style():
    # Try to find a monospaced font
    try:
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        mono_font = 'DejaVu Sans Mono' # A common, good one
        if 'Consolas' in available_fonts: # Often on Windows
            mono_font = 'Consolas'
        elif 'Menlo' in available_fonts: # Often on macOS
            mono_font = 'Menlo'
        elif 'Liberation Mono' in available_fonts:
            mono_font = 'Liberation Mono'
        
        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.monospace'] = mono_font
    except Exception:
        print("Monospaced font not found, using Matplotlib default.")

    plt.rcParams['figure.facecolor'] = '#1E1E2E' # Darker purple/blue
    plt.rcParams['axes.facecolor'] = '#282A3A'    # Slightly lighter dark
    plt.rcParams['axes.edgecolor'] = '#44475A'
    plt.rcParams['axes.labelcolor'] = '#F8F8F2'  # Off-white/Light gray
    plt.rcParams['axes.titlecolor'] = '#BD93F9'  # Light purple for titles
    plt.rcParams['xtick.color'] = '#6272A4'      # Bluish gray for ticks
    plt.rcParams['ytick.color'] = '#6272A4'
    plt.rcParams['grid.color'] = '#44475A'       # Darker grid
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 5
    # Neon-like colors: Cyan, Magenta, Yellow, Green, Orange, Pink
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#50FA7B', '#FF79C6', '#F1FA8C', '#8BE9FD', '#FFB86C', '#FF5555'])

def reset_plot_style():
    plt.rcdefaults() # Reset to Matplotlib defaults

# --- Data Parsing Functions ---
def parse_args_json(output_folder):
    args_file = os.path.join(output_folder, 'args.json')
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            return json.load(f)
    return {}

def parse_training_metrics(output_folder):
    metrics_file = os.path.join(output_folder, 'training_logs', 'train_metrics_log.json')
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    rounds = sorted([int(r) for r in data.keys()])
    metrics = {
        'rounds': rounds,
        'loss_raw': [data[str(r)].get('loss_raw') for r in rounds],
        'avg_tool_calls': [data[str(r)].get('avg_tool_calls', 0) for r in rounds]
    }
    # Filter out None values if a key was missing for some rounds
    valid_indices_loss = [i for i, x in enumerate(metrics['loss_raw']) if x is not None]
    metrics['rounds_loss'] = [metrics['rounds'][i] for i in valid_indices_loss]
    metrics['loss_raw'] = [metrics['loss_raw'][i] for i in valid_indices_loss]

    valid_indices_tools = [i for i, x in enumerate(metrics['avg_tool_calls']) if x is not None]
    metrics['rounds_tools'] = [metrics['rounds'][i] for i in valid_indices_tools]
    metrics['avg_tool_calls'] = [metrics['avg_tool_calls'][i] for i in valid_indices_tools]
    
    return metrics

def parse_evaluation_metrics(output_folder):
    eval_files = sorted(glob.glob(os.path.join(output_folder, 'eval_logs', 'eval_results_round_*.json')))
    if not eval_files:
        return None
    
    eval_data = {
        'rounds': [],
        'win_rates': [],
        'avg_tool_calls': []
    }
    for f_path in eval_files:
        try:
            round_num_str = os.path.basename(f_path).replace('eval_results_round_', '').replace('.json', '')
            round_num = int(round_num_str)
            with open(f_path, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('final_metrics', {})
            if metrics:
                eval_data['rounds'].append(round_num)
                eval_data['win_rates'].append(metrics.get('win_rate'))
                eval_data['avg_tool_calls'].append(metrics.get('avg_tool_calls'))
        except Exception as e:
            print(f"Warning: Could not parse eval file {f_path}: {e}")

    # Sort all collected evaluation data by round number to ensure correct plotting order
    if eval_data['rounds']:
        sorted_indices = sorted(range(len(eval_data['rounds'])), key=lambda k: eval_data['rounds'][k])
        eval_data['rounds'] = [eval_data['rounds'][i] for i in sorted_indices]
        eval_data['win_rates'] = [eval_data['win_rates'][i] if eval_data['win_rates'] else None for i in sorted_indices]
        eval_data['avg_tool_calls'] = [eval_data['avg_tool_calls'][i] if eval_data['avg_tool_calls'] else None for i in sorted_indices]

    # Filter out None values after sorting
    if eval_data['win_rates']:
        valid_indices_wr = [i for i, x in enumerate(eval_data['win_rates']) if x is not None]
        eval_data['rounds_wr'] = [eval_data['rounds'][i] for i in valid_indices_wr]
        eval_data['win_rates'] = [eval_data['win_rates'][i] for i in valid_indices_wr]
    else:
        eval_data['rounds_wr'] = []
        eval_data['win_rates'] = []

    if eval_data['avg_tool_calls']:
        valid_indices_tools = [i for i, x in enumerate(eval_data['avg_tool_calls']) if x is not None]
        eval_data['rounds_tools'] = [eval_data['rounds'][i] for i in valid_indices_tools]
        eval_data['avg_tool_calls'] = [eval_data['avg_tool_calls'][i] for i in valid_indices_tools]
    else:
        eval_data['rounds_tools'] = []
        eval_data['avg_tool_calls'] = []
            
    return eval_data

# --- Plotting Functions ---
def plot_training_loss(data, save_path):
    if not data or not data['rounds_loss']:
        print("No training loss data to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(data['rounds_loss'], data['loss_raw'], marker='o', linestyle='-')
    plt.title('Training Loss Over Rounds', fontsize=16)
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Raw Loss', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training loss plot to {save_path}")

def plot_training_tool_use(data, save_path):
    if not data or not data['rounds_tools']:
        print("No training tool use data to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(data['rounds_tools'], data['avg_tool_calls'], marker='s', linestyle='-')
    plt.title('Average Tool Calls During Training', fontsize=16)
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Average Tool Calls', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training tool use plot to {save_path}")

def plot_eval_win_rate(data, save_path, training_model_name="N/A", compare_model_name="N/A"):
    if not data or not data['rounds_wr']:
        print("No evaluation win rate data to plot.")
        return
    title = f'picoDeepResearch ({training_model_name}) vs {compare_model_name} - Win Rate'
    plt.figure(figsize=(10, 6))
    plt.plot(data['rounds_wr'], data['win_rates'], marker='D', linestyle='-')
    plt.title(title, fontsize=16)
    plt.xlabel('Evaluation Round', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved eval win rate plot to {save_path}")

def plot_eval_tool_use(data, save_path):
    if not data or not data['rounds_tools']:
        print("No evaluation tool use data to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(data['rounds_tools'], data['avg_tool_calls'], marker='X', linestyle='-')
    plt.title('Average Tool Calls During Evaluation', fontsize=16)
    plt.xlabel('Evaluation Round', fontsize=12)
    plt.ylabel('Average Tool Calls', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved eval tool use plot to {save_path}")

# --- HTML Generation ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{ font-family: 'Consolas', 'Menlo', 'DejaVu Sans Mono', monospace; background-color: #1E1E2E; color: #F8F8F2; margin: 0; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: auto; background-color: #282A3A; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }}
        h1, h2, h3 {{ color: #BD93F9; border-bottom: 1px solid #44475A; padding-bottom: 10px; }}
        h1 {{ text-align: center; font-size: 2.5em; }}
        h2 {{ font-size: 1.8em; margin-top: 30px; }}
        h3 {{ font-size: 1.4em; margin-top: 20px; color: #50FA7B; }}
        pre {{ background-color: #1E1E2E; color: #F8F8F2; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #44475A; }}
        .plots img {{ max-width: 100%; height: auto; border-radius: 5px; margin-bottom: 20px; border: 2px solid #44475A; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .tab-container {{ margin-top: 20px; }}
        .tabs {{ display: flex; border-bottom: 2px solid #44475A; }}
        .tab-button {{ background-color: #44475A; color: #F8F8F2; border: none; padding: 10px 20px; cursor: pointer; font-size: 1em; border-radius: 5px 5px 0 0; margin-right: 5px; }}
        .tab-button.active {{ background-color: #6272A4; color: #FFFFFF; }}
        .tab-content {{ display: none; padding: 20px; border: 1px solid #44475A; border-top: none; border-radius: 0 0 5px 5px; background-color: #282A3A; }}
        .tab-content.active {{ display: block; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        li a {{ color: #8BE9FD; text-decoration: none; }}
        li a:hover {{ text-decoration: underline; color: #FF79C6; }}
        .report-item {{ background-color: #1E1E2E; padding: 10px; margin-bottom:10px; border-radius:4px; border-left: 3px solid #50FA7B; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_title}</h1>
        <p style="text-align:center; font-style:italic;">Generated on: {generation_date}</p>

        <h2>Run Arguments</h2>
        <pre>{run_args_html}</pre>

        <h2>Visualizations</h2>
        <div class="plot-grid">
            {plots_html}
        </div>

        <h2>Detailed Reports & Logs</h2>
        <div class="tab-container">
            <div class="tabs">
                <button class="tab-button active" onclick="openTab(event, 'training-summary')">Training Summary</button>
                <button class="tab-button" onclick="openTab(event, 'evaluation-summary')">Evaluation Summary</button>
            </div>

            <div id="training-summary" class="tab-content active">
                <h3>Training Round Reports (PDFs)</h3>
                {training_reports_html}
            </div>

            <div id="evaluation-summary" class="tab-content">
                <h3>Evaluation Round Reports (PDFs & Logs)</h3>
                {evaluation_reports_html}
            </div>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
    </script>
</body>
</html>
"""

def generate_html_report(output_folder, report_filename, plot_paths, run_args, training_pdf_reports, eval_pdf_reports, eval_txt_logs):
    report_title_text = f"picoDeepResearch Run Report: {os.path.basename(output_folder)}"
    generation_date_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_args_html = json.dumps(run_args, indent=4) if run_args else "Not available."

    plots_html_list = []
    for plot_name, plot_path in plot_paths.items():
        if plot_path and os.path.exists(plot_path): # Check if plot was actually generated
             # Relative path for HTML
            relative_plot_path = os.path.join(os.path.basename(os.path.dirname(plot_path)), os.path.basename(plot_path))
            plots_html_list.append(f'''
            <div class="plot-item">
                <h3>{plot_name.replace("_", " ").title()}</h3>
                <img src="{relative_plot_path}" alt="{plot_name}">
            </div>''')
    plots_html_content = "\n".join(plots_html_list)
    if not plots_html_content:
        plots_html_content = "<p>No plots were generated or found.</p>"


    training_reports_list = "<ul>"
    if training_pdf_reports:
        for report_path in training_pdf_reports:
            relative_path = os.path.join("..", "training_logs", os.path.basename(report_path)) # Go up one level from assets
            training_reports_list += f'<li class="report-item"><a href="{relative_path}" target="_blank">{os.path.basename(report_path)}</a></li>'
    else:
        training_reports_list += "<li>No training PDF reports found.</li>"
    training_reports_list += "</ul>"
    
    eval_reports_list = "<ul>"
    if eval_pdf_reports or eval_txt_logs:
        # Combine and sort by round number if possible
        all_eval_files = {}
        for report_path in eval_pdf_reports:
            try:
                round_num = int(os.path.basename(report_path).split('_')[-1].split('.')[0]) # eval_report_round_X.pdf
                key = f"eval_round_{round_num:04d}_pdf" # eval_round_0040_pdf
                all_eval_files[key] = (f'<a href="../eval_logs/{os.path.basename(report_path)}" target="_blank">{os.path.basename(report_path)}</a> (PDF Report)', round_num)
            except:
                 all_eval_files[os.path.basename(report_path) + "_pdf"] = (f'<a href="../eval_logs/{os.path.basename(report_path)}" target="_blank">{os.path.basename(report_path)}</a> (PDF Report)', -1)


        for log_path in eval_txt_logs:
            try: # eval_log_round_X_question_Y.txt
                parts = os.path.basename(log_path).replace('.txt','').split('_')
                round_num = int(parts[3])
                q_num = int(parts[5])
                key = f"eval_round_{round_num:04d}_q_{q_num:03d}_log" # eval_round_0040_q_000_log
                all_eval_files[key] = (f'<a href="../eval_logs/{os.path.basename(log_path)}" target="_blank">{os.path.basename(log_path)}</a> (Detailed Log)', round_num)
            except:
                all_eval_files[os.path.basename(log_path) + "_txt"] = (f'<a href="../eval_logs/{os.path.basename(log_path)}" target="_blank">{os.path.basename(log_path)}</a> (Detailed Log)', -1)

        # Sort by key (which includes round and question number)
        for key in sorted(all_eval_files.keys()):
            eval_reports_list += f'<li class="report-item">{all_eval_files[key][0]}</li>'

    else:
        eval_reports_list += "<li>No evaluation reports or logs found.</li>"
    eval_reports_list += "</ul>"

    html_content = HTML_TEMPLATE.format(
        report_title=report_title_text,
        generation_date=generation_date_text,
        run_args_html=run_args_html,
        plots_html=plots_html_content,
        training_reports_html=training_reports_list,
        evaluation_reports_html=eval_reports_list
    )

    # Save HTML file inside the assets folder
    report_asset_folder = os.path.join(output_folder, "html_report_assets")
    final_html_path = os.path.join(report_asset_folder, report_filename)

    with open(final_html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated HTML report: {final_html_path}")


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots and HTML report for a picoDeepResearch run.")
    parser.add_argument("--output_folder", required=True, help="Path to the run's output directory.")
    parser.add_argument("--report_filename", default="pico_deep_research_report.html", help="Name of the output HTML file.")
    
    args = parser.parse_args()

    # Create assets directory for plots and HTML report
    report_asset_folder = os.path.join(args.output_folder, "html_report_assets")
    os.makedirs(report_asset_folder, exist_ok=True)

    # Apply plot style
    apply_retro_futurism_style()

    # Parse data
    run_args_data = parse_args_json(args.output_folder)
    training_metrics_data = parse_training_metrics(args.output_folder)
    eval_metrics_data = parse_evaluation_metrics(args.output_folder)

    # Plot paths dictionary
    plot_paths = {
        "training_loss": None,
        "training_tool_use": None,
        "evaluation_win_rate": None,
        "evaluation_tool_use": None
    }

    # Generate plots
    if training_metrics_data:
        plot_paths["training_loss"] = os.path.join(report_asset_folder, "training_loss.png")
        plot_training_loss(training_metrics_data, plot_paths["training_loss"])
        
        plot_paths["training_tool_use"] = os.path.join(report_asset_folder, "training_tool_use.png")
        plot_training_tool_use(training_metrics_data, plot_paths["training_tool_use"])

    if eval_metrics_data:
        training_model_name = run_args_data.get("model_name", "TrainingModel")
        # If model_name is a path, take the last part
        if '/' in training_model_name:
            training_model_name = training_model_name.split('/')[-1]
            
        compare_model_name = run_args_data.get("compare_model_name", "CompareModel")
        if '/' in compare_model_name:
            compare_model_name = compare_model_name.split('/')[-1]

        plot_paths["evaluation_win_rate"] = os.path.join(report_asset_folder, "eval_win_rate.png")
        plot_eval_win_rate(eval_metrics_data, plot_paths["evaluation_win_rate"], training_model_name, compare_model_name)
        
        plot_paths["evaluation_tool_use"] = os.path.join(report_asset_folder, "eval_tool_use.png")
        plot_eval_tool_use(eval_metrics_data, plot_paths["evaluation_tool_use"])
    
    # Reset plot style so it doesn't affect other matplotlib uses if this script is imported
    reset_plot_style()
    
    # Collect report file paths
    training_pdfs = sorted(glob.glob(os.path.join(args.output_folder, 'training_logs', 'training_report_round_*.pdf')))
    eval_pdfs = sorted(glob.glob(os.path.join(args.output_folder, 'eval_logs', 'eval_report_round_*.pdf')))
    eval_txts = sorted(glob.glob(os.path.join(args.output_folder, 'eval_logs', 'eval_log_round_*.txt')))


    # Generate HTML report
    generate_html_report(args.output_folder, args.report_filename, plot_paths, run_args_data, training_pdfs, eval_pdfs, eval_txts)

    print("\nReport generation complete.")
    print(f"HTML report and plots saved in: {report_asset_folder}") 