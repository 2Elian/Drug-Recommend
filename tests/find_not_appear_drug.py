#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/7 14:58
# @Author  : lizimo@nuist.edu.cn
# @File    : find_not_appear_drug.py.py
# @Description:
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
import numpy as np

# set chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_jsonl_data(file_path):
    """加载JSONL文件并提取药品信息"""
    drug_counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 提取出院带药列表
                if '出院带药列表' in data and isinstance(data['出院带药列表'], list):
                    for drug in data['出院带药列表']:
                        drug_counter[drug] += 1
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line[:100]}...")
                continue

    return drug_counter


def load_json_data(file_path):
    """加载JSON文件并提取药品信息"""
    drug_counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            for item in data:
                if 'prediction' in item and isinstance(item['prediction'], list):
                    for drug in item['prediction']:
                        drug_counter[drug] += 1
        except json.JSONDecodeError:
            print("错误: 无法解析JSON文件")

    return drug_counter


def plot_drug_frequency(counter, title, filename, top_n=20):
    """绘制药品频率分布图"""
    # 获取前top_n个最常见的药品
    most_common = counter.most_common(top_n)
    drugs = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 条形图
    y_pos = np.arange(len(drugs))
    bars = ax1.barh(y_pos, counts, color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(drugs, fontsize=10)
    ax1.set_xlabel('出现次数', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - 药品频率分布 (前{top_n}名)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()  # 最高的在顶部

    # 在条形上添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', ha='left', va='center', fontsize=9)

    # 饼图 (显示前10名)
    top_10 = most_common[:10]
    pie_drugs = [item[0] for item in top_10]
    pie_counts = [item[1] for item in top_10]

    # 计算其他药品的总数
    other_count = sum(count for _, count in most_common[10:])
    if other_count > 0:
        pie_drugs.append('其他药品')
        pie_counts.append(other_count)

    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_drugs)))
    wedges, texts, autotexts = ax2.pie(pie_counts, labels=pie_drugs, autopct='%1.1f%%',
                                       colors=colors, startangle=90)

    # 美化饼图文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    for text in texts:
        text.set_fontsize(10)

    ax2.set_title(f'{title} - 药品分布比例', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # 打印统计信息
    total_drugs = len(counter)
    total_occurrences = sum(counter.values())
    print(f"{title}统计:")
    print(f"总药品种类: {total_drugs}")
    print(f"总出现次数: {total_occurrences}")
    print(f"平均每种药品出现次数: {total_occurrences / total_drugs:.2f}")
    print("-" * 50)


def compare_drug_sets(jsonl_drugs, json_drugs):
    """比较两个药品集合的差异"""
    jsonl_set = set(jsonl_drugs.keys())
    json_set = set(json_drugs.keys())

    # 在JSON中出现但不在JSONL中出现的药品
    json_only = json_set - jsonl_set
    # 在JSONL中出现但不在JSON中出现的药品
    jsonl_only = jsonl_set - json_set

    print("=" * 60)
    print("药品集合比较结果")
    print("=" * 60)

    print(f"\n在预测JSON中出现但不在原始JSONL中出现的药品 ({len(json_only)}种):")
    if json_only:
        for i, drug in enumerate(sorted(json_only), 1):
            print(f"{i:2d}. {drug}")
    else:
        print("无")

    print(f"\n在原始JSONL中出现但不在预测JSON中出现的药品 ({len(jsonl_only)}种):")
    if jsonl_only:
        for i, drug in enumerate(sorted(jsonl_only), 1):
            print(f"{i:2d}. {drug} (出现次数: {jsonl_drugs[drug]})")
    else:
        print("无")

    # 计算重叠度
    common_drugs = jsonl_set & json_set
    overlap_percentage = len(common_drugs) / len(jsonl_set) * 100

    print(f"\n集合重叠统计:")
    print(f"JSONL总药品数: {len(jsonl_set)}")
    print(f"JSON总药品数: {len(json_set)}")
    print(f"共同药品数: {len(common_drugs)}")
    print(f"重叠度: {overlap_percentage:.2f}%")

    return json_only, jsonl_only


def plot_comparison_chart(jsonl_drugs, json_drugs):
    """绘制两个数据集的比较图表"""
    jsonl_set = set(jsonl_drugs.keys())
    json_set = set(json_drugs.keys())

    # 计算各种集合的大小
    jsonl_only = len(jsonl_set - json_set)
    json_only = len(json_set - jsonl_set)
    common = len(jsonl_set & json_set)

    # 创建韦恩图风格的数据
    categories = ['JSONL独有', '共同药品', 'JSON独有']
    counts = [jsonl_only, common, json_only]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(categories, counts, color=['lightcoral', 'lightsteelblue', 'lightgreen'],
                  alpha=0.7, edgecolor='black')

    ax.set_ylabel('药品种类数量', fontsize=12, fontweight='bold')
    ax.set_title('JSONL与JSON药品集合比较', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('drug_set_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def main():
    # 文件路径 - 请根据实际情况修改
    jsonl_file = 'medical_data.jsonl'  # 替换为你的JSONL文件路径
    json_file = 'predictions.json'  # 替换为你的JSON文件路径

    try:
        # 加载数据
        print("正在加载数据...")
        jsonl_drugs = load_jsonl_data(jsonl_file)
        json_drugs = load_json_data(json_file)

        # 绘制JSONL药品频率图
        print("\n生成JSONL药品频率图...")
        plot_drug_frequency(jsonl_drugs, '原始JSONL数据', 'jsonl_drug_frequency.png')

        # 绘制JSON药品频率图
        print("\n生成JSON药品频率图...")
        plot_drug_frequency(json_drugs, '预测JSON数据', 'json_drug_frequency.png')

        # 比较药品集合
        json_only, jsonl_only = compare_drug_sets(jsonl_drugs, json_drugs)

        # 绘制比较图表
        plot_comparison_chart(jsonl_drugs, json_drugs)

        # 保存详细统计结果
        with open('drug_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("药品分析报告\n")
            f.write("=" * 50 + "\n\n")

            f.write("JSONL数据统计:\n")
            f.write(f"总药品种类: {len(jsonl_drugs)}\n")
            f.write(f"总出现次数: {sum(jsonl_drugs.values())}\n\n")

            f.write("JSON数据统计:\n")
            f.write(f"总药品种类: {len(json_drugs)}\n")
            f.write(f"总出现次数: {sum(json_drugs.values())}\n\n")

            f.write("集合差异分析:\n")
            f.write(f"JSON独有药品数: {len(json_only)}\n")
            f.write(f"JSONL独有药品数: {len(jsonl_only)}\n")

        print("\n分析完成！生成的文件:")
        print("- jsonl_drug_frequency.png: JSONL药品频率图")
        print("- json_drug_frequency.png: JSON药品频率图")
        print("- drug_set_comparison.png: 集合比较图")
        print("- drug_analysis_report.txt: 详细分析报告")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()