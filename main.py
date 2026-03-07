import pandas as pd
import numpy as np
import json
import os


def generate_interactive_dashboard(file_path):
    # read csv
    data = pd.read_csv(file_path, encoding='utf-8')

    # basic clean
    data['code_id'] = data['code'].astype(str).str[:3]
    data['year'] = pd.to_numeric(data['year'], errors='coerce')

    disease_name_map = {
        "A15": "Respiratory TB (Confirmed)",
        "A16": "Respiratory TB (Unconfirmed)",
        "A18": "TB of Other Organs",
        "A19": "Miliary TB",
        "A17": "TB of Nervous System",
        "B16": "Acute Hepatitis B",
        "B15": "Acute Hepatitis A",
        "B18": "Chronic Viral Hepatitis",
        "B37": "Candidiasis"
    }
    need_codes = ["A15", "A16", "A18", "A19", "A17", "B16", "B15", "B18", "B37"]

    # keep target codes
    data = data[data['code_id'].isin(need_codes)].copy()

    num_cols = [
        'Admissions', 'Emergency', 'Mean length of stay', 'Mean time waited',
        'Mean age', 'Age 60-74', 'Age 75+', 'FCE Bed Days'
    ]

    # convert numeric cols
    for col in num_cols:
      if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # group data
    grouped_data = data.groupby(['year', 'code_id'], as_index=False).agg({
        'Admissions': 'sum',
        'Emergency': 'sum',
        'FCE Bed Days': 'sum',
        'Age 60-74': 'sum',
        'Age 75+': 'sum',
        'Mean age': 'mean',
        'Mean time waited': 'mean',
        'Mean length of stay': 'mean'
    })

    grouped_data = grouped_data[grouped_data['Admissions'] > 0].copy()

    # build metrics
    grouped_data['emergency_rate'] = grouped_data['Emergency'] / grouped_data['Admissions']
    grouped_data['intensity'] = grouped_data['Admissions']
    grouped_data['older_share'] = (grouped_data['Age 60-74'] + grouped_data['Age 75+']) / grouped_data['Admissions']
    grouped_data['wait_time'] = grouped_data['Mean time waited']

    final_data = grouped_data.copy()
    metric_cols = ['emergency_rate', 'intensity', 'Mean age', 'wait_time', 'older_share']

    # global z-score
    for metric in metric_cols:
        avg_value = final_data[metric].mean(skipna=True)
        std_value = final_data[metric].std(ddof=0, skipna=True)
        final_data[f'z_{metric}'] = 0 if (pd.isna(std_value) or std_value == 0) else (final_data[metric] - avg_value) / std_value

    # within z-score
    def within_group_z(s):
        std_value = s.std(ddof=0)
        if pd.isna(std_value) or std_value == 0:
            return pd.Series([0] * len(s), index=s.index)
        return (s - s.mean()) / std_value

    final_data['z_intensity_within'] = final_data.groupby('code_id')['intensity'].transform(within_group_z)

    for metric in ['emergency_rate', 'intensity', 'Mean age', 'wait_time', 'older_share']:
        final_data[f'z_{metric}_within'] = final_data.groupby('code_id')[metric].transform(within_group_z)

    # sort heatmap rows
    show_codes = need_codes.copy()
    first_year = 1998
    rank_map = (
        final_data.loc[final_data['year'] == first_year, ['code_id', 'z_intensity_within']]
        .dropna()
        .drop_duplicates(subset=['code_id'])
        .set_index('code_id')['z_intensity_within']
        .to_dict()
    )
    show_codes = sorted(show_codes, key=lambda x: rank_map.get(x, float('inf')))
    show_names = [disease_name_map[x] for x in show_codes if x in disease_name_map]

    year_list = sorted(final_data['year'].dropna().astype(int).unique().tolist())

    # make heatmap data
    heat_data = []
    for i, one_year in enumerate(year_list):
        for j, one_code in enumerate(show_codes):
            temp_value = final_data[(final_data['year'] == one_year) & (final_data['code_id'] == one_code)]['z_intensity_within']
            if not temp_value.empty and pd.notna(temp_value.iloc[0]):
                heat_data.append([i, j, round(temp_value.iloc[0], 2)])

    # make line data
    line_data = []
    for _, row in final_data.iterrows():
      if row['code_id'] in need_codes and pd.notna(row['year']):
            line_data.append({
                "name": disease_name_map[row['code_id']],
                "year": int(row['year']),
                "values": [
                    round(row['z_emergency_rate_within'], 2) if pd.notna(row['z_emergency_rate_within']) else 0,
                    round(row['z_intensity_within'], 2) if pd.notna(row['z_intensity_within']) else 0,
                    round(row['z_Mean age_within'], 2) if pd.notna(row['z_Mean age_within']) else 0,
                    round(row['z_wait_time_within'], 2) if pd.notna(row['z_wait_time_within']) else 0,
                    round(row['z_older_share_within'], 2) if pd.notna(row['z_older_share_within']) else 0
                ]
            })

    # distance to a15
    base_code = "A15"
    feature_cols = [
        'z_emergency_rate_within',
        'z_intensity_within',
        'z_Mean age_within',
        'z_wait_time_within',
        'z_older_share_within'
    ]

    def build_series(code_id):
        temp_data = final_data[final_data["code_id"] == code_id].set_index("year")
        result_list = []
        for one_year in year_list:
            if one_year in temp_data.index:
                one_row = temp_data.loc[one_year]
                one_vec = [one_row[f] if pd.notna(one_row[f]) else np.nan for f in feature_cols]
            else:
                one_vec = [np.nan] * len(feature_cols)
            result_list.append(one_vec)
        return np.array(result_list, dtype=float)

    base_seq = build_series(base_code)

    distance_list = []
    for one_code in need_codes:
        seq = build_series(one_code)
        one_year_distance = np.sqrt(np.nansum((seq - base_seq) ** 2, axis=1))
        valid_mask = ~np.isnan(one_year_distance)
        mean_distance = float(np.nanmean(one_year_distance[valid_mask])) if valid_mask.any() else np.nan
        distance_list.append((one_code, mean_distance))

    distance_map = dict(distance_list)
    sorted_distance_list = [(base_code, distance_map.get(base_code, 0.0))] + sorted(
        [x for x in distance_list if x[0] != base_code],
        key=lambda x: (np.inf if np.isnan(x[1]) else x[1])
    )

    distance_names = [disease_name_map[x] for x, _ in sorted_distance_list]
    distance_values = [
        0.0 if x == base_code else (round(y, 3) if pd.notna(y) else None)
        for x, y in sorted_distance_list
    ]

    metric_list = [
        ('z_emergency_rate_within', 'Urgency'),
        ('z_intensity_within', 'Intensity'),
        ('z_Mean age_within', 'Age Profile'),
        ('z_wait_time_within', 'Wait Time'),
        ('z_older_share_within', 'Elderly Share')
    ]

    # build radar distance
    pivot_map = {}
    for f, _ in metric_list:
        pivot_map[f] = final_data.pivot_table(index='year', columns='code_id', values=f, aggfunc='mean')

    radar_distance_data = []
    for one_code in need_codes:
        one_value_list = []
        for f, _ in metric_list:
            pivot_table = pivot_map[f]
            if (base_code not in pivot_table.columns) or (one_code not in pivot_table.columns):
                one_value_list.append(None)
                continue

            diff_value = (pivot_table[one_code] - pivot_table[base_code]).abs()
            avg_diff = diff_value.mean(skipna=True)
            one_value_list.append(round(float(avg_diff), 3) if pd.notna(avg_diff) else None)

        radar_distance_data.append({
            "name": disease_name_map[one_code],
            "values": one_value_list
        })

    # pack json
    all_json = json.dumps({
        "years": year_list,
        "codeNames": show_names,
        "heatmap": heat_data,
        "radarData": line_data,
        "metrics": ["Urgency", "Intensity", "Age Profile", "Wait Time", "Elderly Share"],
        "distanceNames": distance_names,
        "distanceValues": distance_values,
        "distanceRadarData": radar_distance_data
    }, ensure_ascii=False)

    # build html
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clinical Data Visualization Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
body {{ background-color: #ffffff; color: #1e293b; padding: 20px; font-family: 'Inter', sans-serif; }}
.chart-card {{ border: 1px solid #e2e8f0; border-radius: 12px; background: #f8fafc; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
.matrix-h {{ height: 400px; }}
.bottom-h {{ height: 400px; }}
</style>
</head>
<body>
<header class="mb-8 border-b border-slate-200 pb-4">
<h1 class="text-3xl font-extrabold text-slate-900 tracking-tight">Interactive Analysis: Respiratory TB vs. Comparative Diseases (1998-2023)</h1>
<p class="text-slate-500 mt-2 text-sm tracking-widest font-medium">Course 2 of Research Method</p>
</header>

<div class="chart-card">
<h2 class="text-lg font-bold text-slate-800 mb-4">Macro Trend Matrix <p class="text-slate-500 mt-2 text-sm tracking-widest font-medium">Within-disease Z-Score of Admission Intensity</p></h2>
<div id="matrix-chart" class="matrix-h bg-white rounded-lg"></div>
</div>

<div class="flex flex-col lg:flex-row gap-6">
<div class="lg:w-[40%] chart-card">
<h2 class="text-lg font-bold text-slate-800 mb-4">Annual Disease Profile <p id="radar-subtitle" class="text-slate-600 mt-2 text-sm tracking-widest font-medium"></p></h2>
<div id="fingerprint-radar-chart" class="bottom-h bg-white rounded-lg"></div>
</div>
<div class="lg:w-[60%] chart-card">
<h2 class="text-lg font-bold text-slate-800 mb-4">Metric Evolution <p id="trend-subtitle" class="text-slate-600 mt-2 text-sm tracking-widest font-medium"></p></h2>
<div id="trend-line-chart" class="bottom-h bg-white rounded-lg"></div>
</div>
</div>

<script>
const chartData = {all_json};
const charts = {{}};
const SUB_THEME = ['#990000', '#d95f0e', '#fec44f', '#2b8cbe', '#78c679'];

function initDashboard() {{
    charts.matrix = echarts.init(document.getElementById('matrix-chart'));
    charts.matrix.setOption({{
        tooltip: {{
            position: 'top',
            formatter: p => `Year: ${{chartData.years[p.data[0]]}}<br/>Disease: ${{chartData.codeNames[p.data[1]]}}<br/>Intensity (within-disease Z): ${{Number(p.data[2]).toFixed(2)}}`
        }},
        grid: {{ top: '10%', bottom: '20%', left: 200, right: '80' }},
        xAxis: {{
            type: 'category',
            data: chartData.years,
            name: 'Year',
            nameTextStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }},
            axisLine: {{ show: false }},
            axisTick: {{ show: false }},
            splitLine: {{ show: false }},
            axisLabel: {{ interval: 1, fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        yAxis: {{
            type: 'category',
            data: chartData.codeNames,
            name: 'Disease',
            nameTextStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }},
            axisLine: {{ show: false }},
            axisTick: {{ show: false }},
            splitLine: {{ show: false }},
            axisLabel: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        visualMap: {{
            min: -2,
            max: 2,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '0%',
            color: ['#fff7bc', '#fec44f', '#d95f0e', '#990000'],
            text: ['High', 'Low'],
            textStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }},
            formatter: function (value) {{ return Number(value).toFixed(2); }}
        }},
        series: [{{
            name: 'Intensity',
            type: 'heatmap',
            data: chartData.heatmap
        }}]
    }});

    charts.trend = echarts.init(document.getElementById('trend-line-chart'));
    charts.radar = echarts.init(document.getElementById('fingerprint-radar-chart'));

    charts.matrix.on('click', function (params) {{
        updateTrend(chartData.codeNames[params.data[1]]);
    }});

    const DEFAULT_DISEASE = 'Respiratory TB (Confirmed)';
    const defaultName = chartData.codeNames.includes(DEFAULT_DISEASE)
        ? DEFAULT_DISEASE
        : chartData.codeNames[0];

    updateTrend(defaultName);
    initDistance();
}}

function updateTrend(diseaseName) {{
    document.getElementById('trend-subtitle').innerText = "Evolution: " + diseaseName;
    const rows = chartData.radarData.filter(d => d.name === diseaseName);

    charts.trend.setOption({{
        backgroundColor: '#ffffff',
        tooltip: {{ trigger: 'axis' }},
        grid: {{ left: 30, right: 70, top: 50, bottom: 70, containLabel: true }},
        legend: {{
            bottom: 5,
            data: chartData.metrics,
            textStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        xAxis: {{
            type: 'category',
            data: chartData.years,
            boundaryGap: false,
            axisLine: {{ show: true, onZero: false }},
            axisTick: {{ show: true, alignWithLabel: true }},
            name: 'year',
            nameLocation: 'end',
            nameTextStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }},
            axisLabel: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        yAxis: {{
            type: 'value',
            name: 'Z-Score',
            nameTextStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }},
            axisLine: {{ show: true }},
            axisTick: {{ show: true }},
            splitLine: {{ show: true }},
            axisLabel: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        series: chartData.metrics.map((m, i) => ({{
            name: m,
            type: 'line',
            smooth: 0.2,
            lineStyle: {{ width: 1.5, color: SUB_THEME[i] }},
            itemStyle: {{ color: SUB_THEME[i] }},
            data: chartData.years.map(y => {{
                const d = rows.find(r => r.year === y);
                return d ? d.values[i] : null;
            }})
        }}))
    }}, true);
}}

function initDistance() {{
    document.getElementById('radar-subtitle').innerText =
        "Metric-wise distance to Respiratory TB (Confirmed)";

    const items = (chartData.distanceRadarData || []).filter(
        d => d.name !== "Respiratory TB (Confirmed)"
    );

    let vmax = 0;
    items.forEach(it => {{
        (it.values || []).forEach(v => {{
            if (v !== null && v !== undefined && !isNaN(v)) {{
                vmax = Math.max(vmax, v);
            }}
        }});
    }});
    vmax = vmax === 0 ? 1 : +(vmax * 1.15).toFixed(3);

    const seriesData = items.map((it, idx) => ({{
        name: it.name,
        value: (it.values || []).map(v => (v === null || v === undefined) ? 0 : v),
        lineStyle: {{ width: 1.2, color: SUB_THEME[idx % SUB_THEME.length], opacity: 0.85 }},
        itemStyle: {{ color: SUB_THEME[idx % SUB_THEME.length] }},
        areaStyle: {{ opacity: (it.name === "Respiratory TB (Confirmed)") ? 0.08 : 0.03 }}
    }}));

    charts.radar.setOption({{
        backgroundColor: '#ffffff',
        tooltip: {{
            trigger: 'item',
            formatter: (p) => {{
                const arr = p.value || [];
                const lines = chartData.metrics.map((m, i) => {{
                    const v = arr[i];
                    return `${{m}}: ${{Number(v).toFixed(3)}}`;
                }});
                return `${{p.name}}<br/>` + lines.join('<br/>');
            }}
        }},
        legend: {{
            type: 'plain',
            left: 'center',
            bottom: 0,
            width: '90%',
            itemGap: 14,
            itemWidth: 10,
            itemHeight: 10,
            data: items.map(d => d.name),
            textStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }}
        }},
        radar: {{
            center: ['50%', '42%'],
            indicator: chartData.metrics.map(m => ({{ name: m, max: vmax, min: 0 }})),
            radius: '58%',
            name: {{ textStyle: {{ fontFamily: 'Arial', fontSize: 12, color: '#000000' }} }},
            splitLine: {{ show: true }},
            splitArea: {{ show: true }}
        }},
        series: [{{
            type: 'radar',
            data: seriesData
        }}]
    }}, true);
}}

initDashboard();
window.onresize = () => Object.values(charts).forEach(c => c.resize());
</script>
</body>
</html>
    """

    # save file
    output_file = "figure.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(output_file)


if __name__ == "__main__":
    data_path = 'data.csv'
    generate_interactive_dashboard(data_path)

    """REFERENCE
        The usage of API, Implementation of functions such as chart configuration and click event linkage:
        ECharts API and Chart Configuration Documentation https://echarts.apache.org/handbook
    """
