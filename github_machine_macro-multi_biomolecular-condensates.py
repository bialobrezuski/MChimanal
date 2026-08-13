# -*- coding: utf-8 -*-
"""
Last updated: Mon, Oct 01, 2025, 07:00
@Author: Michal Kacper Bialobrzewski

First publication release of the Automated Dual-Channel Fluorescence Confocal Image Analysis Workflow for Fiji/ImageJ (MChimanal).

This release corresponds to software version 1.0.0 archived in RepOD: https://doi.org/10.18150/2K3PB9

Please cite the following paper when using or adapting this code: https://doi.org/10.64898/2026.07.16.738963

Who is it for?
Designed for bench scientists and imaging specialists who need reproducible, high-throughput quantification of two-color fluorescence data—especially in the context of biomolecular condensate research. It's also ideal for computational biologists seeking ready-to-analyze tables and figures without the hassle of manually operating ImageJ/Fiji.

Setup Instructions: (coding done in utf-8)

a. Install the bundled Fiji.app 1.54p (https://fiji.sc/) and add PTBIOP via the Fiji updater.
b. Before analyzing condensates, enable the following measurement options in Fiji: Area, Standard Deviation, Min & Max, Area Fraction, and Perimeter.
In Analyze Particles, check: Display Results, Clear Results, Summarize, Add to Manager, Overlay.
c. Specify the path to your Fiji executable in lines 232, 1024, and 1045, e.g.:'C:/../fiji-win64/Fiji.app/ImageJ-win64.exe'
d. Set the working directory in line 1103, e.g.:'C:/../'
e. Choose thresholding models from the 19 available in Fiji/ImageJ by editing line 1104.
f. To simplify image filenames, define removable patterns in line 39, e.g. 'pH 7', '150 mM NaCl'.
"""

import os
import cv2
import numpy as np
import subprocess
import shutil
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 3_000_000_000      
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import re
import logging
from pathlib import Path
import gc

def shorten_file_name(file_name):
    patterns_to_remove = [
        'ph 7-2', 
        '150 mM NaCl'
    ]
    new_name = file_name
    for pattern in patterns_to_remove:
        new_name = new_name.replace(pattern, '')
    new_name = re.sub(r'\s+', ' ', new_name).strip()
    new_name = re.sub(r'[<>:"/\\|?*]', '', new_name)
    return new_name

def shorten_file_names_in_folder(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        return
    for file_path in folder.iterdir():
        if file_path.is_dir():
            continue
        original = file_path.name
        new_name = shorten_file_name(original)
        if new_name == original:
            continue
        new_path = folder / new_name
        base, ext = new_path.stem, new_path.suffix
        counter = 1
        while new_path.exists():
            new_path = folder / f"{base}_{counter}{ext}"
            counter += 1
        file_path.rename(new_path)

def prepare_data(output_csv_paths):
    X = []
    y = []
    labels = []
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi
            if 'StdDev' not in df.columns:
                print(f"Column 'StdDev' missing in file: {csv_path}")
                continue
            df['Diam from Area'] = 2 * np.sqrt(df['Area'] / np.pi)
            df['Error Diam from Area'] = (1 / np.sqrt(df['Area'] * np.pi)) * df['StdDev']
            X.append(df[['Area', 'Diameter', 'Diam from Area']].mean().values)
            y.append(model)
            labels.append(model)
        else:
            print(f"File {csv_path} does not exist.")
    if not X:
        print("No valid data was found. Returning None.")
        return None, None, None
    X = np.array(X)
    y_encoded = pd.factorize(np.array(y))[0]
    return X, y_encoded, labels

def run_imagej_analysis(image_path, output_dir, models):

    import os, subprocess, textwrap

    image_path = image_path.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    macro = textwrap.dedent(f"""
    setBatchMode(true);
    setOption("DisableUndo", true);
    call("ij.Prefs.set", "png.compression", 1);
    run("Close All");

    open("{image_path}");
    if (bitDepth!=16) run("16-bit");
    run("Split Channels");

    titles = getList("image.titles");
    c1Title=""; c2Title="";
    for (i=0; i<titles.length; i++) {{
        selectWindow(titles[i]);
        if (indexOf(titles[i],"C1")!=-1 || indexOf(titles[i],"Ch1")!=-1) {{ rename("C1"); c1Title="C1"; }}
        else if (indexOf(titles[i],"C2")!=-1 || indexOf(titles[i],"Ch2")!=-1) {{ rename("C2"); c2Title="C2"; }}
    }}
    if (c1Title=="" || c2Title=="") {{
        titles = getList("image.titles");
        if (titles.length>=2) {{
            selectWindow(titles[0]); rename("C1"); c1Title="C1";
            selectWindow(titles[1]); rename("C2"); c2Title="C2";
        }} else {{
            exit("Two channels not found!");
        }}
    }}

    function autoMinMax() {{
        getStatistics(a, m, minV, maxV, sd, hist);
        if (isNaN(minV) || isNaN(maxV)) return newArray(minV, maxV);
        bins = hist.length;

        iMin = 0;
        while (iMin < bins-1 && hist[iMin]==0) iMin++;
        iMax = bins-1;
        while (iMax > 0 && hist[iMax]==0) iMax--;

        cut = floor(bins*0.002);
        if (iMin < cut) iMin = cut;
        if (iMax > bins-cut-1) iMax = bins-cut-1;

        binW = (maxV-minV)/bins;
        pMin = minV + iMin*binW;
        pMax = minV + iMax*binW;
        if (pMax<=pMin) {{ pMin=minV; pMax=maxV; }}
        return newArray(pMin,pMax);
    }}

    function saveHistCSV(path) {{
        getHistogram(vals, counts, 256);
        f = File.open(path);
        print(f, "Value,Count");
        for (k=0; k<vals.length; k++) print(f, vals[k] + "," + counts[k]);
        File.close(f);
    }}

    // C2 is the reference
    selectWindow("C2");
    mm = autoMinMax(); c2min=mm[0]; c2max=mm[1];
    setMinAndMax(c2min,c2max); run("Apply LUT");
    run("8-bit","scale");
    run("Gaussian Blur...", "sigma=2");
    run("Smooth"); run("Despeckle");
    saveAs("png", "{output_dir}/{name}_C2_original.png");
    saveHistCSV("{output_dir}/C2-intensity-histogram.csv");
    run("Duplicate...", "title=Processed_C2");

    // C1 based on the reference
    selectWindow("C1");
    setMinAndMax(c2min,c2max); run("Apply LUT");
    run("8-bit","scale");
    run("Gaussian Blur...", "sigma=2");
    run("Smooth"); run("Despeckle");
    saveAs("png", "{output_dir}/{name}_C1_original.png");
    saveHistCSV("{output_dir}/C1-intensity-histogram.csv");
    """)

    for m in models:
        macro += textwrap.dedent(f"""
        // -------- {m} ----------
        selectWindow("Processed_C2");
        run("Duplicate...", "title=Processed_{m}");
        selectWindow("Processed_{m}");
        setAutoThreshold("{m}");
        run("Convert to Mask"); run("Make Binary"); run("Watershed");

        run("Analyze Particles...", "size=0.1-Infinity show=Overlay exclude clear add");
        roiManager("Show All with labels");
        saveAs("png", "{output_dir}/{name}_threshold_{m}.png");
        saveAs("Results", "{output_dir}/results_{m}.csv");
        run("Clear Results"); roiManager("Reset");

        run("Analyze Particles...", "size=0.1-Infinity show=Overlay clear add");
        roiManager("Show All with labels");
        saveAs("png", "{output_dir}/{name}_edges_{m}.png");
        saveAs("Results", "{output_dir}/edges_{m}.csv");
        run("Clear Results"); roiManager("Reset");

        close("Processed_{m}");
        """)

    macro += """
    selectWindow("Processed_C2"); close();
    run("Close All");
    setBatchMode(false);
    """

    macro_path = os.path.join(output_dir, "macro_analysis.ijm").replace("\\", "/")
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write(macro)

    created = [
        os.path.join(output_dir, f"{name}_C1_original.png"),
        os.path.join(output_dir, "C1-intensity-histogram.csv"),
        os.path.join(output_dir, f"{name}_C2_original.png"),
        os.path.join(output_dir, "C2-intensity-histogram.csv"),
        macro_path
    ]
    for m in models:
        created += [
            os.path.join(output_dir, f"{name}_threshold_{m}.png"),
            os.path.join(output_dir, f"results_{m}.csv"),
            os.path.join(output_dir, f"{name}_edges_{m}.png"),
            os.path.join(output_dir, f"edges_{m}.csv")
        ]

    try:
        res = subprocess.run(
            ["C:/../fiji-win64/Fiji.app/ImageJ-win64.exe", "-batch", macro_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=600
        )
        if res.returncode != 0 or "Macro Error" in res.stderr:
            print(res.stderr.strip())
    except subprocess.TimeoutExpired:
        print("Fiji exceeded the 600 s time limit.")

    return created

def round_significant(value, sig_digits=2):
    if value == 0:
        return 0
    return round(value, -int(np.floor(np.log10(abs(value))) - (sig_digits - 1)))

def round_dataframe_to_significant(df, columns, sig_digits=2):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round_significant(x, sig_digits))
    return df

def calculate_statistics_and_save(output_csv_paths, output_file_path):
    data = []
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi

            median_area = df['Area'].median()
            q1_area = df['Area'].quantile(0.25)
            q3_area = df['Area'].quantile(0.75)
            median_diameter = df['Diameter'].median()
            q1_diameter = df['Diameter'].quantile(0.25)
            q3_diameter = df['Diameter'].quantile(0.75)
            median_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).median()
            q1_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.25)
            q3_diam_from_area = (2 * np.sqrt(df['Area'] / np.pi)).quantile(0.75)
            count = len(df)

            data.append({
                'Model': model,
                'Median Area': round_significant(median_area),
                'Q1 Area': round_significant(q1_area),
                'Q3 Area': round_significant(q3_area),
                'Median Diameter': round_significant(median_diameter),
                'Q1 Diameter': round_significant(q1_diameter),
                'Q3 Diameter': round_significant(q3_diameter),
                'Median Diam from Area': round_significant(median_diam_from_area),
                'Q1 Diam from Area': round_significant(q1_diam_from_area),
                'Q3 Diam from Area': round_significant(q3_diam_from_area),
                'Count': count,
                'Sqrt Count': round_significant(np.sqrt(count))
            })
    all_models_df = pd.DataFrame(data)

    with pd.ExcelWriter(output_file_path) as writer:
        all_models_df.to_excel(writer, sheet_name='Model Statistics', index=False)

        overall_stats = {
            'Median Area': round_significant(all_models_df['Median Area'].median()),
            'Q1 Area': round_significant(all_models_df['Q1 Area'].median()),
            'Q3 Area': round_significant(all_models_df['Q3 Area'].median()),
            'Median Diameter': round_significant(all_models_df['Median Diameter'].median()),
            'Q1 Diameter': round_significant(all_models_df['Q1 Diameter'].median()),
            'Q3 Diameter': round_significant(all_models_df['Q3 Diameter'].median()),
            'Median Diam from Area': round_significant(all_models_df['Median Diam from Area'].median()),
            'Q1 Diam from Area': round_significant(all_models_df['Q1 Diam from Area'].median()),
            'Q3 Diam from Area': round_significant(all_models_df['Q3 Diam from Area'].median()),
            'Counts': round_significant(all_models_df['Count'].mean()),
            'STD Counts': round_significant(all_models_df['Count'].std())
        }
        pd.DataFrame([overall_stats]).to_excel(writer, sheet_name='Overall Statistics', index=False)

def calculate_edge_statistics_and_save(edge_csv_paths, output_file_path):
    data = []
    for model, csv_path in edge_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi

            median_area = df['Area'].median()
            q1_area = df['Area'].quantile(0.25)
            q3_area = df['Area'].quantile(0.75)
            median_diameter = df['Diameter'].median()
            q1_diameter = df['Diameter'].quantile(0.25)
            q3_diameter = df['Diameter'].quantile(0.75)
            diam_from_area_series = 2 * np.sqrt(df['Area'] / np.pi)
            median_diam_from_area = diam_from_area_series.median()
            q1_diam_from_area = diam_from_area_series.quantile(0.25)
            q3_diam_from_area = diam_from_area_series.quantile(0.75)
            count = len(df)

            data.append({
                'Model': model,
                'Median Area': round_significant(median_area),
                'Q1 Area': round_significant(q1_area),
                'Q3 Area': round_significant(q3_area),
                'Median Diameter': round_significant(median_diameter),
                'Q1 Diameter': round_significant(q1_diameter),
                'Q3 Diameter': round_significant(q3_diameter),
                'Median Diam from Area': round_significant(median_diam_from_area),
                'Q1 Diam from Area': round_significant(q1_diam_from_area),
                'Q3 Diam from Area': round_significant(q3_diam_from_area),
                'Count': count,
                'Sqrt Count': round_significant(np.sqrt(count))
            })
    all_models_df = pd.DataFrame(data)

    with pd.ExcelWriter(output_file_path) as writer:
        all_models_df.to_excel(writer, sheet_name='Model Statistics', index=False)

        overall_stats = {
            'Median Area': round_significant(all_models_df['Median Area'].median()),
            'Q1 Area': round_significant(all_models_df['Q1 Area'].median()),
            'Q3 Area': round_significant(all_models_df['Q3 Area'].median()),
            'Median Diameter': round_significant(all_models_df['Median Diameter'].median()),
            'Q1 Diameter': round_significant(all_models_df['Q1 Diameter'].median()),
            'Q3 Diameter': round_significant(all_models_df['Q3 Diameter'].median()),
            'Median Diam from Area': round_significant(all_models_df['Median Diam from Area'].median()),
            'Q1 Diam from Area': round_significant(all_models_df['Q1 Diam from Area'].median()),
            'Q3 Diam from Area': round_significant(all_models_df['Q3 Diam from Area'].median()),
            'Counts': round_significant(all_models_df['Count'].mean()),
            'STD Counts': round_significant(all_models_df['Count'].std())
        }
        pd.DataFrame([overall_stats]).to_excel(writer, sheet_name='Overall Statistics', index=False)

def merge_results_to_excel(image_name, output_csv_paths, models, output_excel_path, stats_file_path, edge_stats_file_path):
    with pd.ExcelWriter(output_excel_path) as writer:
        stats_df = pd.read_excel(stats_file_path, sheet_name='Model Statistics')
        edge_stats_df = pd.read_excel(edge_stats_file_path, sheet_name='Model Statistics')

        for model in models:
            csv_path = output_csv_paths.get(model, None)
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                    df['Diameter'] = df['Perim.'] / np.pi
                df['Diam from Area'] = 2 * np.sqrt(df['Area'] / np.pi)
                df.to_excel(writer, sheet_name=model, index=False)

        summary_data = []
        for model in models:
            model_stats = stats_df[stats_df['Model'] == model]
            if not model_stats.empty:
                summary_data.append({
                    'Model': model,
                    'Median Area': round_significant(model_stats['Median Area'].values[0]),
                    'Q1 Area': round_significant(model_stats['Q1 Area'].values[0]),
                    'Q3 Area': round_significant(model_stats['Q3 Area'].values[0]),
                    'Median Diameter': round_significant(model_stats['Median Diameter'].values[0]),
                    'Q1 Diameter': round_significant(model_stats['Q1 Diameter'].values[0]),
                    'Q3 Diameter': round_significant(model_stats['Q3 Diameter'].values[0]),
                    'Median Diam from Area': round_significant(model_stats['Median Diam from Area'].values[0]),
                    'Q1 Diam from Area': round_significant(model_stats['Q1 Diam from Area'].values[0]),
                    'Q3 Diam from Area': round_significant(model_stats['Q3 Diam from Area'].values[0]),
                    'Count': model_stats['Count'].values[0],
                    'Sqrt Count': round_significant(np.sqrt(model_stats['Count'].values[0]))
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

def plot_statistics(stats_file_path, edge_stats_file_path, output_plot_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    stats_df = pd.read_excel(stats_file_path, sheet_name='Model Statistics')
    overall_stats = pd.read_excel(stats_file_path, sheet_name='Overall Statistics').iloc[0]

    edge_stats_df = pd.read_excel(edge_stats_file_path, sheet_name='Model Statistics')
    edge_overall_stats = pd.read_excel(edge_stats_file_path, sheet_name='Overall Statistics').iloc[0]

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18
    axis_label_fontsize = 18
    tick_label_fontsize = 16

    count_sd = stats_df['Count'].std()
    overall_count = overall_stats['Counts']

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.errorbar(stats_df['Model'], stats_df['Median Area'],
                 yerr=[stats_df['Median Area'] - stats_df['Q1 Area'], stats_df['Q3 Area'] - stats_df['Median Area']],
                 fmt='o', color='blue', label='Median Area', markersize=8)
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Median Area'],
                 yerr=[edge_stats_df['Median Area'] - edge_stats_df['Q1 Area'], edge_stats_df['Q3 Area'] - edge_stats_df['Median Area']],
                 fmt='s', color='orange', label='Edge Median Area', markersize=8)
    plt.axhline(y=overall_stats['Median Area'], color='r', linestyle='--', label='Overall Median Area')
    plt.axhline(y=overall_stats['Median Area'] + (overall_stats['Q3 Area'] - overall_stats['Median Area']), color='gray', linestyle='--', label='+1Q Area')
    plt.axhline(y=overall_stats['Median Area'] - (overall_stats['Median Area'] - overall_stats['Q1 Area']), color='gray', linestyle='--')
    plt.axhline(y=overall_stats['Median Area'] + 2 * (overall_stats['Q3 Area'] - overall_stats['Median Area']), color='black', linestyle='--', label='+2Q Area')
    plt.axhline(y=overall_stats['Median Area'] - 2 * (overall_stats['Median Area'] - overall_stats['Q1 Area']), color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Area (µm²)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    
    plt.subplot(1, 3, 2)
    plt.errorbar(stats_df['Model'], stats_df['Median Diameter'],
                 yerr=[stats_df['Median Diameter'] - stats_df['Q1 Diameter'], stats_df['Q3 Diameter'] - stats_df['Median Diameter']],
                 fmt='o', color='blue', label='Median Diameter', markersize=8)
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Median Diameter'],
                 yerr=[edge_stats_df['Median Diameter'] - edge_stats_df['Q1 Diameter'], edge_stats_df['Q3 Diameter'] - edge_stats_df['Median Diameter']],
                 fmt='s', color='orange', label='Edge Median Diameter', markersize=8)
    plt.axhline(y=overall_stats['Median Diameter'], color='r', linestyle='--', label='Overall Median Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] + (overall_stats['Q3 Diameter'] - overall_stats['Median Diameter']), color='gray', linestyle='--', label='+1Q Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] - (overall_stats['Median Diameter'] - overall_stats['Q1 Diameter']), color='gray', linestyle='--')
    plt.axhline(y=overall_stats['Median Diameter'] + 2 * (overall_stats['Q3 Diameter'] - overall_stats['Median Diameter']), color='black', linestyle='--', label='+2Q Diameter')
    plt.axhline(y=overall_stats['Median Diameter'] - 2 * (overall_stats['Median Diameter'] - overall_stats['Q1 Diameter']), color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Perimeter-derived diameter (µm)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    
    plt.subplot(1, 3, 3)
    plt.errorbar(stats_df['Model'], stats_df['Count'],
                 yerr=count_sd, fmt='o', color='blue', label='Counts', markersize=8)
    plt.errorbar(edge_stats_df['Model'], edge_stats_df['Count'],
                 yerr=edge_stats_df['Count'].std(), fmt='s', color='orange', label='Edge Counts', markersize=8)
    plt.axhline(y=overall_count, color='r', linestyle='--', label='Overall Median Count')
    plt.axhline(y=overall_count + count_sd, color='gray', linestyle='--', label='+1SD Count')
    plt.axhline(y=overall_count - count_sd, color='gray', linestyle='--')
    plt.axhline(y=overall_count + 2 * count_sd, color='black', linestyle='--', label='+2SD Count')
    plt.axhline(y=overall_count - 2 * count_sd, color='black', linestyle='--')
    plt.xlabel('Threshold Models', fontsize=axis_label_fontsize, labelpad=20)
    plt.ylabel('Counts (a.u.)', fontsize=axis_label_fontsize, labelpad=20)
    plt.xticks(rotation=90, fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=len(labels), fontsize='medium', bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def predict_best_threshold_model(output_csv_paths):
    data = []
    for model, csv_path in output_csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Diameter' not in df.columns and 'Perim.' in df.columns:
                df['Diameter'] = df['Perim.'] / np.pi
            if 'Area' not in df.columns:
                print(f"Column 'Area' missing in file: {csv_path}")
                continue
            df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
            if df['Area'].isnull().all():
                print(f"All 'Area' values are NaN after conversion in file: {csv_path}")
                continue
            diam_from_area = 2 * np.sqrt(df['Area'] / np.pi)
            data.append({
                'Model': model,
                'Median Area': df['Area'].median(),
                'Q1 Area': df['Area'].quantile(0.25),
                'Q3 Area': df['Area'].quantile(0.75),
                'Median Diameter': df['Diameter'].median(),
                'Q1 Diameter': df['Diameter'].quantile(0.25),
                'Q3 Diameter': df['Diameter'].quantile(0.75),
                'Diam from Area': diam_from_area.median(),
                'Q1 Diam from Area': diam_from_area.quantile(0.25),
                'Q3 Diam from Area': diam_from_area.quantile(0.75),
                'Count': len(df)
            })
        else:
            print(f"File {csv_path} does not exist.")
    all_models_df = pd.DataFrame(data)
    if all_models_df.empty:
        print("No valid data found for prediction.")
        return [], [], [], None

    overall_median_area = all_models_df['Median Area'].median()
    overall_median_diameter = all_models_df['Median Diameter'].median()
    overall_median_diam_from_area = all_models_df['Diam from Area'].median()
    overall_median_count = all_models_df['Count'].median()

    sd_area = all_models_df['Median Area'].std()
    sd_diameter = all_models_df['Median Diameter'].std()
    sd_diam_from_area = all_models_df['Diam from Area'].std()
    sd_count = all_models_df['Count'].std()

    def classify_model(row):
        within_1sd = (
            abs(row['Median Area'] - overall_median_area) <= sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= sd_count
        )
        if within_1sd:
            return "Within 1 SD"
        within_2sd = (
            abs(row['Median Area'] - overall_median_area) <= 2 * sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= 2 * sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= 2 * sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= 2 * sd_count
        )
        if within_2sd:
            return "Within 2 SD"
        within_3sd = (
            abs(row['Median Area'] - overall_median_area) <= 3 * sd_area and
            abs(row['Median Diameter'] - overall_median_diameter) <= 3 * sd_diameter and
            abs(row['Diam from Area'] - overall_median_diam_from_area) <= 3 * sd_diam_from_area and
            abs(row['Count'] - overall_median_count) <= 3 * sd_count
        )
        if within_3sd:
            return "Within 3 SD"
        return "Outside 3 SD"

    all_models_df['Model Category'] = all_models_df.apply(classify_model, axis=1)

    best_models_within_1sd = all_models_df[all_models_df['Model Category'] == 'Within 1 SD']['Model'].tolist()
    models_within_2sd = all_models_df[all_models_df['Model Category'] == 'Within 2 SD']['Model'].tolist()
    models_within_3sd = all_models_df[all_models_df['Model Category'] == 'Within 3 SD']['Model'].tolist()

    return best_models_within_1sd, models_within_2sd, models_within_3sd, {
        'Overall Mean Area': overall_median_area,
        'STD Area': sd_area,
        'Overall Mean Diameter': overall_median_diameter,
        'STD Diameter': sd_diameter,
        'Overall Mean Diam from Area': overall_median_diam_from_area,
        'STD Diam from Area': sd_diam_from_area,
        'Overall Mean Count': overall_median_count,
        'STD Count': sd_count
    }

def save_predictions_to_excel(best_models_within_1sd, models_within_2sd, models_within_3sd, model_stats, statistics_file_path, output_file_path):
    if model_stats is None:
        print("No model statistics available. Skipping predictions saving.")
        return

    stats_df = pd.read_excel(statistics_file_path, sheet_name='Model Statistics')
    overall_sd_count = stats_df['Count'].std()

    overall_stats = {
        'Overall Mean Area': round_significant(model_stats['Overall Mean Area'], 2),
        'STD Area': round_significant(model_stats['STD Area'], 2),
        'Overall Mean Diameter': round_significant(model_stats['Overall Mean Diameter'], 2),
        'STD Diameter': round_significant(model_stats['STD Diameter'], 2),
        'Overall Mean Diam from Area': round_significant(model_stats['Overall Mean Diam from Area'], 2),
        'STD Diam from Area': round_significant(model_stats['STD Diam from Area'], 2),
        'Overall Mean Count': round_significant(model_stats['Overall Mean Count'], 2),
        'STD Count': round_significant(overall_sd_count, 2)
    }

    predictions_data = {
        'Model Category': [], 'Model Name': [],
        'Median Area': [], 'Q1 Area': [], 'Q3 Area': [],
        'Median Diameter': [], 'Q1 Diameter': [], 'Q3 Diameter': [],
        'Median Diam from Area': [], 'Q1 Diam from Area': [], 'Q3 Diam from Area': [],
        'Count': [], 'Sqrt Count': [], 'STD Count': []
    }

    def get_model_stats(model_name):
        model_row = stats_df[stats_df['Model'] == model_name]
        if not model_row.empty:
            return model_row.iloc[0]
        return None

    for cat, lst in [('Within 1 SD', best_models_within_1sd),
                     ('Within 2 SD', models_within_2sd),
                     ('Within 3 SD', models_within_3sd)]:
        for model in lst:
            model_stats_row = get_model_stats(model)
            if model_stats_row is not None:
                predictions_data['Model Category'].append(cat)
                predictions_data['Model Name'].append(model)
                predictions_data['Median Area'].append(round_significant(model_stats_row['Median Area'], 2))
                predictions_data['Q1 Area'].append(round_significant(model_stats_row['Q1 Area'], 2))
                predictions_data['Q3 Area'].append(round_significant(model_stats_row['Q3 Area'], 2))
                predictions_data['Median Diameter'].append(round_significant(model_stats_row['Median Diameter'], 2))
                predictions_data['Q1 Diameter'].append(round_significant(model_stats_row['Q1 Diameter'], 2))
                predictions_data['Q3 Diameter'].append(round_significant(model_stats_row['Q3 Diameter'], 2))
                predictions_data['Median Diam from Area'].append(round_significant(model_stats_row.get('Median Diam from Area', 0), 2))
                predictions_data['Q1 Diam from Area'].append(round_significant(model_stats_row.get('Q1 Diam from Area', 0), 2))
                predictions_data['Q3 Diam from Area'].append(round_significant(model_stats_row.get('Q3 Diam from Area', 0), 2))
                predictions_data['Count'].append(round_significant(model_stats_row['Count'], 2))
                predictions_data['Sqrt Count'].append(round_significant(np.sqrt(model_stats_row['Count']), 2))
                predictions_data['STD Count'].append(round_significant(overall_sd_count, 2))

    with pd.ExcelWriter(output_file_path) as writer:
        pd.DataFrame([overall_stats]).to_excel(writer, sheet_name='Overall Statistics', index=False)
        pd.DataFrame(predictions_data).to_excel(writer, sheet_name='Shewhart Predictions', index=False)

    print("✓ Predictions of best threshold models saved.")

def generate_grid(output_dir, threshold_models, grid_size=5, thumbnail_size=(200, 200),
                 padding=30, font_size=40, frame_color=(0, 0, 0), text_color=(0, 0, 0),
                 background_color=(255, 255, 255), frame_thickness=3, side_margin=150, top_margin=150, bottom_margin=150,
                 text_padding=30, scale_factor=1):
    image_info = []
    original_image_path = None
    for file in os.listdir(output_dir):
        if file.endswith("_C2_original.png"):
            original_image_path = os.path.join(output_dir, file)
            break
    if original_image_path:
        image_info.append((original_image_path, 'Original'))

    for model in threshold_models:
        img_files = [f for f in os.listdir(output_dir) if f.endswith(f"_threshold_{model}.png")]
        for img_file in img_files:
            image_path = os.path.join(output_dir, img_file)
            image_info.append((image_path, model))

    num_images = len(image_info)
    num_rows = (num_images + grid_size - 1) // grid_size

    try:
        font = ImageFont.truetype("arial.ttf", font_size * scale_factor)
    except IOError:
        font = ImageFont.load_default()

    thumbnail_width, thumbnail_height = thumbnail_size
    grid_img_width = (thumbnail_width + padding) * grid_size - padding + 2 * side_margin
    grid_img_height = (thumbnail_height + padding + font_size + text_padding) * num_rows + top_margin + bottom_margin

    grid_img_width_scaled = grid_img_width * scale_factor
    grid_img_height_scaled = grid_img_height * scale_factor

    grid_img = Image.new('RGB', (grid_img_width_scaled, grid_img_height_scaled), background_color)
    draw = ImageDraw.Draw(grid_img)

    for i, (img_path, label) in enumerate(image_info):
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Cannot open image: {img_path}")
            continue

        img = add_padding(img, thumbnail_size, background_color)
        img = img.resize(thumbnail_size, Image.Resampling.LANCZOS)

        row = i // grid_size
        col = i % grid_size
        x_offset = col * (thumbnail_width + padding) + side_margin
        y_offset = row * (thumbnail_height + padding + font_size + text_padding) + top_margin

        x_offset_scaled = x_offset * scale_factor
        y_offset_scaled = y_offset * scale_factor

        thumbnail_width_scaled = thumbnail_width * scale_factor
        thumbnail_height_scaled = thumbnail_height * scale_factor
        frame_thickness_scaled = frame_thickness * scale_factor
        text_padding_scaled = text_padding * scale_factor
        font_size_scaled = font_size * scale_factor

        if isinstance(font, ImageFont.FreeTypeFont):
            try:
                font_scaled = ImageFont.truetype("arial.ttf", font_size_scaled)
            except IOError:
                font_scaled = font
        else:
            font_scaled = font

        frame = Image.new('RGB', (thumbnail_width_scaled + 2 * frame_thickness_scaled, 
                                  thumbnail_height_scaled + 2 * frame_thickness_scaled), frame_color)
        frame.paste(img.resize((thumbnail_width_scaled, thumbnail_height_scaled), Image.Resampling.LANCZOS), 
                   (frame_thickness_scaled, frame_thickness_scaled))
        grid_img.paste(frame, (x_offset_scaled, y_offset_scaled))

        text_bbox = draw.textbbox((0, 0), label, font=font_scaled)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset_scaled + (thumbnail_width_scaled - text_width) // 2
        text_y = y_offset_scaled + thumbnail_height_scaled + text_padding_scaled

        draw.text((text_x + int(2 * scale_factor), text_y + int(2 * scale_factor)), label, font=font_scaled, fill=(255, 255, 255))
        draw.text((text_x, text_y), label, font=font_scaled, fill=text_color)

    grid_img_path = os.path.join(output_dir, 'grid_threshold_summary.png')
    grid_img.save(grid_img_path, format='PNG', dpi=(300, 300))
    print("✓ Grid image saved.")

    plt.figure(figsize=(grid_img_width_scaled / 100, grid_img_height_scaled / 100), dpi=300)
    plt.imshow(np.array(grid_img))
    plt.axis('off')
    plt.close()

def add_padding(img, target_size, background_color=(255, 255, 255)):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    if img_ratio > target_ratio:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, background_color)
    paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    new_img.paste(img, paste_position)
    return new_img

def move_files_to_folder(files, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for file in files:
        if os.path.exists(file):
            try:
                dest = os.path.join(folder_name, os.path.basename(file))
                shutil.move(file, dest)
            except Exception as e:
                print(f"Failed to move file {file} to {folder_name}: {e}")

def generate_intensity_histogram(output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from PIL import Image
    import os

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18
    axis_label_fontsize = 18
    tick_label_fontsize = 16

    def process_channel_data(channel_name, csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'Value' not in df.columns or 'Count' not in df.columns:
                print(f"Expected columns 'Value' and 'Count' not found in histogram data for {channel_name}.")
                return None, None
            values = np.repeat(df['Value'], df['Count'].astype(int))
            values = values[values >= 0]
            if values.size == 0:
                print(f"No valid intensity data to plot for {channel_name}.")
                return None, None

            fluorescence_excel_path = os.path.join(output_dir, f'fluorescence-intensities-{channel_name}.xlsx')
            histogram_df = df[['Value', 'Count']].copy()
            histogram_df.columns = ['values', 'count']
            histogram_df['V⋅c'] = histogram_df['values'] * histogram_df['count']
            histogram_df['sum V⋅c'] = np.nan
            if len(histogram_df) >= 1:
                histogram_df.at[0, 'sum V⋅c'] = histogram_df['V⋅c'].sum()
            histogram_df.to_excel(fluorescence_excel_path, index=False)

            return values, df
        except Exception as e:
            print(f"Error processing {channel_name} data: {e}")
            return None, None

    def determine_gradient_color(image_path):
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
                red_pixels = (image_np[:, :, 0] > 200) & (image_np[:, :, 1] < 50) & (image_np[:, :, 2] < 50)
                green_pixels = (image_np[:, :, 1] > 200) & (image_np[:, :, 0] < 50) & (image_np[:, :, 2] < 50)
                yellow_pixels = (image_np[:, :, 0] > 200) & (image_np[:, :, 1] > 200) & (image_np[:, :, 2] < 50)
                if np.any(red_pixels):
                    return 'orangered'
                elif np.any(green_pixels):
                    return 'lime'
                elif np.any(yellow_pixels):
                    return 'yellow'
            return 'white'
        except Exception as e:
            print(f"Error determining gradient color: {e}")
            return 'white'

    channel_image_paths = {
        'C1': next((os.path.join(output_dir, file) for file in os.listdir(output_dir) if "C1_original.png" in file), None),
        'C2': next((os.path.join(output_dir, file) for file in os.listdir(output_dir) if "C2_original.png" in file), None)
    }

    gradient_colors = {channel: determine_gradient_color(image_path) for channel, image_path in channel_image_paths.items()}
    color_maps = {channel: LinearSegmentedColormap.from_list('CustomScale', [(0, 'black'), (1, color)]) for channel, color in gradient_colors.items()}

    channel_data = {'C1': os.path.join(output_dir, 'C1-intensity-histogram.csv'), 
                    'C2': os.path.join(output_dir, 'C2-intensity-histogram.csv')}
    data_values = {}
    for channel, path in channel_data.items():
        if os.path.exists(path):
            data_values[channel] = process_channel_data(channel, path)
        else:
            print(f"Histogram CSV missing for {channel}: {path}")
            data_values[channel] = (None, None)
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 7), gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
    sns.set(style='whitegrid')
    
    for ax, (channel, (values, df)) in zip(axes, data_values.items()):
        if values is None or df is None:
            ax.axis('off')
            continue
        
        ax.hist(values, bins=256, color='white', edgecolor='black', alpha=0.55, log=True)
        ax.set_xlim(0, 260)
        ax.set_ylim(1000, 1e7)
        ax.grid(True)
        ax.set_xlabel('Intensity Value', fontsize=axis_label_fontsize)
        ax.set_ylabel('Frequency (log scale)', fontsize=axis_label_fontsize)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)
        ax.set_title(f'{channel} Intensity Distribution Histogram', fontsize=20)
        
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        plt.subplots_adjust(bottom=0.3)
        
        axins = inset_axes(ax, width="100%", height="5%", loc='lower center', bbox_to_anchor=(0, -0.25, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        axins.imshow(gradient, aspect='auto', cmap=color_maps[channel], extent=[0, 255, 0, 1], origin='lower')
        axins.axis('off')
        axins.set_xticks([0, 255])
        axins.set_xticklabels(['Min', 'Max'], fontsize=axis_label_fontsize)

    histogram_plot_path = os.path.join(output_dir, 'intensity_distribution_histograms.png')
    plt.savefig(histogram_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_colocalization(image_path: str, out_dir: str) -> None:

    import os, subprocess, pandas as pd, textwrap, re, shutil
    from PIL import Image
    import numpy as np
    from pathlib import Path

    GRID_THR = 0.25          
    SAVE_BOTH_TABLES = True  

    def gridiness(png_path: str) -> float:

        try:
            im = Image.open(png_path).convert("L")
            arr = np.array(im, dtype=np.int16)
            thr = 8  # próg std
            col_flat = (arr.std(axis=0) < thr).mean()
            row_flat = (arr.std(axis=1) < thr).mean()
            return float((col_flat + row_flat) / 2)
        except Exception:
            return 1.0

    def parse_block(txt: str) -> dict:

        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        params = {}
        i = 0
        while i < len(lines):
            L = lines[i]
            if L.startswith("Pearson"):
                i += 1
                while i < len(lines) and not lines[i].startswith("r="): i += 1
                if i < len(lines): params["Pearson's r"] = lines[i].split("=",1)[1].strip()
            elif L.startswith("Spearman"):
                i += 1
                while i < len(lines) and not lines[i].startswith("r="): i += 1
                if i < len(lines): params["Spearman's rho"] = lines[i].split("=",1)[1].strip()
            elif L.startswith("Manders' Coefficients (original)"):
                i += 1
                while i < len(lines) and not lines[i].startswith("M1="): i += 1
                if i < len(lines): params["Manders M1 (orig)"] = lines[i].split("=",1)[1].split()[0]
                i += 1
                if i < len(lines) and lines[i].startswith("M2="):
                    params["Manders M2 (orig)"] = lines[i].split("=",1)[1].split()[0]
            elif L.startswith("Manders' Coefficients (using threshold"):
                i += 1
                while i < len(lines) and not lines[i].startswith("M1="): i += 1
                if i < len(lines): params["Manders M1 (thr)"] = lines[i].split("=",1)[1].split()[0]
                i += 1
                if i < len(lines) and lines[i].startswith("M2="):
                    params["Manders M2 (thr)"] = lines[i].split("=",1)[1].split()[0]
            elif L.startswith("Overlap Coefficient:") and "Using thresholds" not in L:
                i += 1
                while i < len(lines) and not lines[i].startswith("r="): i += 1
                if i < len(lines): params["Overlap r (orig)"] = lines[i].split("=",1)[1].strip()
            elif L.startswith("Using thresholds"):
                i += 1
                while i < len(lines) and not lines[i].startswith("Overlap Coefficient"): i += 1
                i += 1
                if i < len(lines) and lines[i].startswith("r="):
                    params["Overlap r (thr)"] = lines[i].split("=",1)[1].strip()
            elif L.startswith("r^2"):
                i += 1
                while i < len(lines) and not lines[i].startswith("k1="): i += 1
                if i < len(lines): params["k1 (orig)"] = lines[i].split("=",1)[1].strip()
                i += 1
                if i < len(lines) and lines[i].startswith("k2="):
                    params["k2 (orig)"] = lines[i].split("=",1)[1].strip()
            elif L.startswith("Li's Intensity correlation"):
                i += 1
                while i < len(lines) and "ICQ" not in lines[i]: i += 1
                if i < len(lines): params["ICQ"] = lines[i].split(":",1)[1].strip()
            elif L.startswith("Cytofluorogram's parameters"):
                i += 1
                while i < len(lines) and not lines[i].startswith("a:"): i += 1
                if i < len(lines): params["Cytofluorogram a"] = lines[i].split(":",1)[1].strip()
                i += 1
                if i < len(lines) and lines[i].startswith("b:"):
                    params["Cytofluorogram b"] = lines[i].split(":",1)[1].strip()
            elif L.startswith("Area ROI"):
                m = re.search(r"Area tot\s*=\s*([\d\.Ee+-]+)", L)
                if m: params["Total ROI Area (µm²/pixel)"] = m.group(1)
            elif L.startswith("Area Measurements"):
                i += 1
                if i < len(lines):
                    parts = re.findall(r"Area A=([\d\.Ee+-]+).*Area B=([\d\.Ee+-]+)", lines[i])
                    if parts:
                        a,b = parts[0]
                        params["Area A"] = a
                        params["Area B"] = b
            elif L.startswith("Area Overlap"):
                params["Overlap Area"] = L.split("=",1)[1].strip()
            i += 1
        return params

    def write_table(params: dict, path_csv: str):
        df = pd.DataFrame([params])
        df.to_csv(path_csv, index=False)
        df.to_excel(path_csv.replace(".csv", ".xlsx"), index=False)

    xlsx = os.path.join(out_dir, "threshold_model_predictions.xlsx")
    best_model = "Default" if not os.path.exists(xlsx) else pd.read_excel(xlsx, sheet_name=1, header=None).iloc[1, 1]

    img  = image_path.replace("\\", "/")
    od   = out_dir.replace("\\", "/")
    name = os.path.splitext(os.path.basename(image_path))[0]

    def make_macro(tag: str, use_fix: bool) -> str:
        prefix = "set " if use_fix else ""
        bins   = 128 if use_fix else 256
        sw     = 0.0  if use_fix else 0.2
        return textwrap.dedent(f"""
        setBatchMode(true);
        setOption("DisableUndo", true);
        call("ij.Prefs.set", "png.compression", 1);
        run("Close All");

        open("{img}");
        if (bitDepth!=16) run("16-bit");
        run("Split Channels");

        titles = getList("image.titles");
        for (i=0;i<titles.length;i++) {{
            selectWindow(titles[i]);
            if (indexOf(titles[i],"C1")!=-1 || indexOf(titles[i],"Ch1")!=-1) rename("C1");
            else if (indexOf(titles[i],"C2")!=-1 || indexOf(titles[i],"Ch2")!=-1) rename("C2");
            else close();
        }}

        run("Merge Channels...", "c1=C2 c2=C1 create keep");
        rename("Merge");
        saveAs("png","{od}/{name}_Merge.png");
        close("Merge");

        print("###BEGIN_{tag}###");
        run("BIOP JACoP",
            "channel_a=2 channel_b=1 " +
            "threshold_for_channel_a=[{best_model}] threshold_for_channel_b=[{best_model}] " +
            "manual_threshold_a=0 manual_threshold_b=0 " +
            "get_pearsons get_spearmanrank get_manders get_overlap get_li_ica get_fluorogram " +
            "costes_block_size=5 costes_number_of_shuffling=100 " +
            "{prefix}fluorogram_bins={bins} fluorogram_min=0 fluorogram_max=255 " +
            "xmin_costes_graph=-1 xmax_costes_graph=1 stroke_width={sw}");
        print("###END_{tag}###");

        function saveFluo(path){{
            ok=false;
            setBatchMode(false);
            for (t=0;t<120;t++) {{
                wait(100);
                titles=getList("image.titles");
                for (j=0;j<titles.length;j++) {{
                    if (indexOf(titles[j],"Report")!=-1 || indexOf(titles[j],"Fluorogram")!=-1 || indexOf(titles[j],"Cytofluorogram")!=-1){{
                        selectWindow(titles[j]);
                        if (getWidth()>2500) run("Scale...", "x=0.5 y=0.5 interpolation=Bicubic");
                        saveAs("png", path);
                        ok=true;
                        break;
                    }}
                }}
                if (ok) break;
            }}
            setBatchMode(true);
        }}
        saveFluo("{od}/{name}_fluorogram_{tag}.png");

        run("Close All");
        setBatchMode(false);
        """)

    ijm0 = os.path.join(od, "macro_coloc_VAR0.ijm").replace("\\","/")
    Path(ijm0).write_text(make_macro("VAR0", use_fix=False), encoding="utf-8")

    res0 = subprocess.run(
        ["C:/../fiji-win64/Fiji.app/ImageJ-win64.exe", "-batch", ijm0],
        capture_output=True, text=True, timeout=1800
    )
    raw0 = res0.stdout

    raw_log_path = os.path.join(out_dir, f"{name}_colocalization-results.txt")
    Path(raw_log_path).write_text(raw0, encoding="utf-8")

    fluo0 = os.path.join(out_dir, f"{name}_fluorogram_VAR0.png")
    g0 = gridiness(fluo0) if os.path.exists(fluo0) else 1.0
    need_fix = (not os.path.exists(fluo0)) or g0 > GRID_THR

    chosen_tag = "VAR0"
    g1 = None

    if need_fix:
        ijm1 = os.path.join(od, "macro_coloc_VAR1.ijm").replace("\\","/")
        Path(ijm1).write_text(make_macro("VAR1", use_fix=True), encoding="utf-8")

        res1 = subprocess.run(
            ["C:/../fiji-win64/Fiji.app/ImageJ-win64.exe", "-batch", ijm1],
            capture_output=True, text=True, timeout=1800
        )
        raw1 = res1.stdout
        with open(raw_log_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + raw1)

        fluo1 = os.path.join(out_dir, f"{name}_fluorogram_VAR1.png")
        g1 = gridiness(fluo1) if os.path.exists(fluo1) else 1.0

        if g1 < g0:  # wybierz lepszy
            chosen_tag = "VAR1"


    full_txt = Path(raw_log_path).read_text(encoding="utf-8")


    m_var0 = re.search(r"###BEGIN_VAR0###(.*?)###END_VAR0###", full_txt, flags=re.S)
    m_var1 = re.search(r"###BEGIN_VAR1###(.*?)###END_VAR1###", full_txt, flags=re.S)


    m_chosen = re.search(rf"###BEGIN_{chosen_tag}###(.*?)###END_{chosen_tag}###", full_txt, flags=re.S)
    if m_chosen:
        block_txt = m_chosen.group(1)
    else:

        blocks = [b for b in full_txt.split("**************************************************") if "Pearson" in b]
        block_txt = blocks[0] if blocks else full_txt

    params = parse_block(block_txt)

    # meta‑info
    params["variant_used"]       = chosen_tag
    params["fluorogram_used"]    = f"{name}_fluorogram_{chosen_tag}.png"
    params["gridiness_VAR0"]     = g0
    params["gridiness_VAR1"]     = g1
    params["bins_used"]          = 256 if chosen_tag == "VAR0" else 128
    params["stroke_width_used"]  = 0.2 if chosen_tag == "VAR0" else 0.0
    params["GRID_THR"]           = GRID_THR

    table_csv = os.path.join(out_dir, f"{name}_colocalization-table.csv")
    write_table(params, table_csv)

    if SAVE_BOTH_TABLES:
        if m_var0:
            write_table(parse_block(m_var0.group(1)),
                        table_csv.replace(".csv", "_VAR0.csv"))
        if m_var1:
            write_table(parse_block(m_var1.group(1)),
                        table_csv.replace(".csv", "_VAR1.csv"))

    best_src = os.path.join(out_dir, f"{name}_fluorogram_{chosen_tag}.png")
    if os.path.exists(best_src):
        shutil.copyfile(best_src, os.path.join(out_dir, f"{name}_Fluorogram.png"))

def main():
    folder_path = 'C:/../...'
    threshold_models = [
        "Default", "Huang", "Intermodes", "IsoData", "Li",
        "MaxEntropy", "Moments", "Otsu", "RenyiEntropy"
    ]

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.lsm') and 'exp' in file_name:
            image_path = os.path.join(folder_path, file_name)
            output_dir = os.path.join(folder_path,
                                      os.path.splitext(file_name)[0].strip())
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n──────────────────────────────────────────────")
            print(f"Processing file: {file_name}")
            print("Thresholding the image …")
            try:
                run_imagej_analysis(image_path, output_dir, threshold_models)
                print("✓ ImageJ thresholding analysis completed.")
            except Exception as e:
                print(f"✗ ImageJ analysis failed: {e}")
                continue

            print("Generating intensity histograms …")
            generate_intensity_histogram(output_dir)
            print("✓ Histograms generated.")

            output_csv = {m: os.path.join(output_dir, f"results_{m}.csv")
                          for m in threshold_models}
            edge_csv   = {m: os.path.join(output_dir, f"edges_{m}.csv")
                          for m in threshold_models}

            X, _, _ = prepare_data(output_csv)
            if X is None:
                print("No data – skipping further steps.")
            else:
                stats_xlsx      = os.path.join(output_dir, 'statistics.xlsx')
                edge_stats_xlsx = os.path.join(output_dir, 'statistics_edges.xlsx')

                calculate_statistics_and_save(output_csv,      stats_xlsx)
                calculate_edge_statistics_and_save(edge_csv,   edge_stats_xlsx)

                generate_grid(output_dir, threshold_models, scale_factor=2)

                merge_results_to_excel(file_name, output_csv, threshold_models,
                                       os.path.join(output_dir,'summary_threshold_results.xlsx'),
                                       stats_xlsx, edge_stats_xlsx)

                merge_results_to_excel(file_name, edge_csv, threshold_models,
                                       os.path.join(output_dir,'summary_threshold_edges_results.xlsx'),
                                       stats_xlsx, edge_stats_xlsx)

                plot_statistics(stats_xlsx, edge_stats_xlsx,
                                os.path.join(output_dir,'statistics_plots.png'))

                best1, best2, best3, stats_pred = predict_best_threshold_model(output_csv)
                save_predictions_to_excel(best1, best2, best3, stats_pred,
                                             stats_xlsx,
                                             os.path.join(output_dir,
                                                          'threshold_model_predictions.xlsx'))
                
                print("✓ Statistics and Shewhart predictions done.")

            try:
                print("Running colocalization (BIOP JACoP) and generating high-quality scatter-plots …")
                run_colocalization(image_path, output_dir)
                print("✓ Colocalization finished.")
            except Exception as e:
                print(f"✗ Colocalization failed: {e}")

            edges_img_folder = os.path.join(output_dir, 'edges analysis')
            edges_csv_folder = os.path.join(output_dir, 'edges csv')
            macra_folder     = os.path.join(output_dir, 'macra')
            csvs_folder      = os.path.join(output_dir, 'csvs')

            os.makedirs(edges_img_folder, exist_ok=True)
            os.makedirs(edges_csv_folder, exist_ok=True)
            os.makedirs(macra_folder, exist_ok=True)
            os.makedirs(csvs_folder, exist_ok=True)

            move_files_to_folder(
                [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.lower().endswith('.png') and 'edges_' in f],
                edges_img_folder
            )

            move_files_to_folder(
                [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.lower().endswith('.csv') and f.lower().startswith('edges_')],
                edges_csv_folder
            )

            move_files_to_folder(
                [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                 if f.lower().endswith('.ijm')],
                macra_folder
            )

            threshold_csvs = [
                f for f in os.listdir(output_dir)
                if f.lower().endswith('.csv') and f.lower().startswith('results_')
            ]
            coloc_csvs = [
                f for f in os.listdir(output_dir)
                if f.lower().endswith('.csv') and 'colocalization-' in f.lower()
            ]
            hist_csvs = [
                'C1-intensity-histogram.csv',
                'C2-intensity-histogram.csv'
            ]
            all_csvs = threshold_csvs + coloc_csvs + hist_csvs
            move_files_to_folder(
                [os.path.join(output_dir, f) for f in all_csvs],
                csvs_folder
            )

            shorten_file_names_in_folder(output_dir)

            new_out = os.path.join(folder_path,
                                   shorten_file_name(os.path.basename(output_dir)))
            if output_dir != new_out:
                counter = 1
                tmp = new_out
                while os.path.exists(tmp):
                    counter += 1
                    tmp = f"{new_out}-{counter}"
                shutil.move(output_dir, tmp)
                output_dir = tmp

            gc.collect()
            print(f"✔ Memory cleaned.")
            print(f"✔ Done: {file_name}")

if __name__ == "__main__":
    main()