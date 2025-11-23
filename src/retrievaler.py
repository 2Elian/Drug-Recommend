# 覆盖比策略权重0.5，诊断匹配0.2，TF-IDF 0.3

class EnsembleDrugRecommender:
    def __init__(self, retrieval):
        self.retrieval = retrieval
        self.strategies = [
            self.coverage_based_strategy,
            self.diagnosis_match_strategy,
            self.tfidf_based_strategy,  # 新增方法2
        ]
    
    def recommend(self, test_diagnoses, min_frequency=1):
        """集成推荐"""
        base_recommendations = self.retrieval.get_drugs_by_diagnoses(test_diagnoses, min_frequency=min_frequency)
        
        if not base_recommendations:
            return []
        
        # 🔥 完善方法1：在基础推荐中直接计算并存储cover_ratio
        self._enhance_recommendations_with_cover_ratio(base_recommendations, test_diagnoses)
        
        # 🔥 完善方法1：先按cover_ratio和frequency双重排序
        pre_sorted_recommendations = self._pre_sort_by_cover_ratio_and_frequency(base_recommendations)
        
        # 应用多种策略
        strategy_scores = {}
        for strategy in self.strategies:
            scores = strategy(pre_sorted_recommendations, test_diagnoses)
            for drug, score in scores.items():
                if drug not in strategy_scores:
                    strategy_scores[drug] = []
                strategy_scores[drug].append(score)
        
        # 集成得分（加权平均，给覆盖比更高权重）
        final_scores = {}
        for drug, scores in strategy_scores.items():
            # 加权平均：覆盖比策略权重0.5，诊断匹配0.2，TF-IDF 0.3 目前最好的结果AVG_Jaccard: 0.552691 AVG_Precision: 0.805081 AVG_Recall: 0.584996
            # 与测试集精度损失占比--> Jaccar是4.18 P是2.59  R是3.2
            weights = [0.5, 0.2, 0.3]
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            final_scores[drug] = weighted_score
        
        # 排序和选择
        sorted_drugs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择策略：基于得分差距的动态选择
        selected_drugs = self.dynamic_selection(sorted_drugs)
        
        return selected_drugs
    
    def _enhance_recommendations_with_cover_ratio(self, recommendations, test_diagnoses):
        """完善方法1：为每个药物计算并存储覆盖比"""
        for drug_info in recommendations:
            drug_diseases = set(drug_info["treating_diagnoses"])
            target_diseases = set(test_diagnoses)
            drug_info["cover_ratio"] = len(drug_diseases & target_diseases) / len(target_diseases)
    
    def _pre_sort_by_cover_ratio_and_frequency(self, recommendations):
        """完善方法1：先按覆盖比和频率双重排序"""
        return sorted(
            recommendations,
            key=lambda x: (-x["cover_ratio"], -x["frequency"])  # 覆盖比降序，频率降序
        )
    
    def coverage_based_strategy(self, recommendations, test_diagnoses):
        """基于覆盖率的策略 - 使用存储的cover_ratio"""
        scores = {}
        for drug_info in recommendations:
            scores[drug_info['drug']] = drug_info["cover_ratio"]  # 直接使用预计算的覆盖比
        return scores
    
    def diagnosis_match_strategy(self, recommendations, test_diagnoses):
        """基于诊断匹配的策略"""
        scores = {}
        for drug_info in recommendations:
            # 检查是否匹配主要诊断
            main_match = 1.0 if test_diagnoses and test_diagnoses[0] in drug_info['treating_diagnoses'] else 0.0
            scores[drug_info['drug']] = main_match
        return scores
    
    def tfidf_based_strategy(self, recommendations, test_diagnoses):
        """方法2：TF-IDF-like策略"""
        import math
        
        # 获取总诊断数量（需要从知识图谱中查询，这里简化处理）
        total_diagnoses_count = 1000  # 假设总诊断数量，实际应该从KG获取
        
        scores = {}
        for drug_info in recommendations:
            # 计算诊断数量（该药物关联的诊断总数）
            diagnosis_count = len(set(drug_info['treating_diagnoses']))
            
            # 计算TF-IDF得分
            tf = drug_info['frequency']  # 词频（频率）
            idf = math.log(total_diagnoses_count / max(1, diagnosis_count))  # 逆诊断频率
            tfidf_score = tf * idf
            
            # 归一化到0-1范围
            scores[drug_info['drug']] = min(1.0, tfidf_score / 10)  # 假设最大得分约10
        
        return scores
    
    def dynamic_selection(self, sorted_drugs):
        """动态选择药物数量"""
        if not sorted_drugs:
            return []
        
        scores = [score for _, score in sorted_drugs]
        
        # 寻找得分差距较大的点
        threshold_index = 0
        for i in range(1, len(scores)):
            if scores[i-1] - scores[i] > 0.2:  # 得分差距阈值
                threshold_index = i
                break
        
        # 如果没有明显差距，选择前3个
        if threshold_index == 0:
            threshold_index = min(3, len(sorted_drugs))
        
        return [drug for drug, _ in sorted_drugs[:threshold_index]]
# retrieval.py - 药物检索专用代码
from neo4j import GraphDatabase
import json
from typing import List, Dict, Any
from datetime import datetime
import argparse, json, sys
from typing import Dict, List, Tuple, Set
def load_jsonl_data(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"{e}")
    return data
class MedicalKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()

class MedicalDrugRetrieval:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg
    
    def get_drugs_by_diagnoses(self, diagnoses: List[str], min_frequency: int = 1):
        with self.kg.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Disease)-[r:TREATED_WITH]->(dr:Drug)
                WHERE d.name IN $diagnoses
                WITH dr.name AS drug_name, 
                     COUNT(DISTINCT d) AS covered_diagnoses,
                     SUM(r.weight) AS total_frequency,
                     COLLECT(DISTINCT d.name) AS treating_diagnoses
                WHERE covered_diagnoses >= $min_cover
                RETURN drug_name, total_frequency, covered_diagnoses, treating_diagnoses
                ORDER BY total_frequency DESC, covered_diagnoses DESC
                """,
                diagnoses=diagnoses, min_cover=min_frequency
            )
            
            recommendations = []
            for record in result:
                recommendations.append({
                    "drug": record["drug_name"],
                    "frequency": record["total_frequency"],
                    "covered_diagnoses": record["covered_diagnoses"],
                    "treating_diagnoses": record["treating_diagnoses"]
                })
            return recommendations
    
    def get_detailed_recommendations(self, diagnoses: List[str]):
        results = {}
        all_drugs = self.get_drugs_by_diagnoses(diagnoses, min_frequency=1)
        results['all_related_drugs'] = all_drugs
        if len(diagnoses) > 1:
            multi_diagnosis_drugs = self.get_drugs_by_diagnoses(
                diagnoses, min_frequency=2
            )
            results['multi_diagnosis_drugs'] = multi_diagnosis_drugs
        perfect_drugs = self.get_drugs_by_diagnoses(
            diagnoses, min_frequency=len(diagnoses)
        )
        results['perfect_coverage_drugs'] = perfect_drugs
        
        return results

def get_recommend(min_frequency: int = 1, test_file: str = None, save_path: str = None):
    NEO4J_URI = "bolt://172.16.107.15:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "MyStrongPassword123"
    kg = MedicalKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    retrieval = MedicalDrugRetrieval(kg)
    
    # 创建集成推荐器
    ensemble_recommender = EnsembleDrugRecommender(retrieval)
    
    save = []
    try:
        datas = load_jsonl_data(test_file)
        for data in datas:
            drug_id = data.get('就诊标识')
            test_diagnoses = data.get('出院诊断')
            
            if not test_diagnoses:
                print(f'{drug_id}的出院诊断是空的')
                save.append({'ID': drug_id, "prediction": []})
                continue
            
            # 使用增强版推荐
            recommendations = ensemble_recommender.recommend(test_diagnoses, min_frequency=min_frequency)
            
            save.append({
                'ID': drug_id,
                "prediction": recommendations
            })
        
        # 保存结果
        output_file = f"{save_path}/enhanced_drug_recommendations-{min_frequency}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save, f, ensure_ascii=False, indent=2)
        
        print(f"增强推荐结果已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
    finally:
        if kg:
            kg.close()


def load_gt_jsonl(path: str, id_field: str, label_field: str, lower: bool) -> Dict[str, Set[str]]:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
            obj = json.loads(line)
            if id_field not in obj:
                raise KeyError(f"[GT] 行{ln}缺少字段: {id_field}")
            if label_field not in obj:
                raise KeyError(f"[GT] 行{ln}缺少字段: {label_field}")
            _id = str(obj[id_field])
            labels = obj[label_field]
            if isinstance(labels, str):
                labels = [labels]
            if not isinstance(labels, list):
                raise TypeError(f"[GT] 行{ln}的 {label_field} 不是list/str")
            items = [str(x).strip() for x in labels if str(x).strip()!=""]
            if lower: items = [x.lower() for x in items]
            gt[_id] = set(items)
    return gt

def load_pred(path: str, id_field: str, label_field: str, lower: bool) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred = {}
    if isinstance(data, dict):
        # 形如 {ID: [..], ...}
        for _id, lst in data.items():
            items = [str(x).strip() for x in (lst or []) if str(x).strip()!=""]
            if lower: items = [x.lower() for x in items]
            pred[str(_id)] = items
    elif isinstance(data, list):
        # 形如 [{"ID":..,"prediction":[..]}, ...]
        for i, obj in enumerate(data, 1):
            if id_field not in obj:
                raise KeyError(f"[PRED] 第{i}个样本缺少字段: {id_field}")
            if label_field not in obj:
                raise KeyError(f"[PRED] 第{i}个样本缺少字段: {label_field}")
            _id = str(obj[id_field])
            lst = obj[label_field] or []
            if isinstance(lst, str):
                lst = [lst]
            if not isinstance(lst, list):
                raise TypeError(f"[PRED] 第{i}个样本的 {label_field} 不是list/str")
            items = [str(x).strip() for x in lst if str(x).strip()!=""]
            if lower: items = [x.lower() for x in items]
            pred[_id] = items
    else:
        raise TypeError("[PRED] 不支持的JSON结构：应为对象或数组")

    return pred

def jaccard(p: Set[str], y: Set[str]) -> float:
    if not p and not y: return 1.0
    return len(p & y) / max(1, len(p | y))

def precision(p: Set[str], y: Set[str]) -> float:
    if not p:
        return 1.0 if not y else 0.0
    return len(p & y) / len(p)

def recall(p: Set[str], y: Set[str]) -> float:
    if not y:
        return 1.0 if not p else 0.0
    return len(p & y) / len(y)

def f1(p: Set[str], y: Set[str]) -> float:
    P = precision(p, y)
    R = recall(p, y)
    if P + R == 0: return 0.0
    return 2 * P * R / (P + R)

def evaluate(gt: Dict[str, Set[str]], pred: Dict[str, List[str]]) -> Tuple[float,float,float,float,float]:
    ids = list(gt.keys())
    miss = [i for i in ids if i not in pred]
    extra = [i for i in pred.keys() if i not in gt]
    if miss:
        print(f"[警告] 预测缺少 {len(miss)} 个ID（将按空预测处理）示例: {miss[:5]}", file=sys.stderr)
    if extra:
        print(f"[提示] 预测包含 {len(extra)} 个不在GT中的ID（将忽略）示例: {extra[:5]}", file=sys.stderr)

    J_list, P_list, R_list, F1_list = [], [], [], []
    for _id, y in gt.items():
        p = set(pred.get(_id, []))  # 缺失按空预测
        J_list.append(jaccard(p, y))
        P_list.append(precision(p, y))
        R_list.append(recall(p, y))
        F1_list.append(f1(p, y))

    AVG_J = sum(J_list)/len(J_list)
    AVG_P = sum(P_list)/len(P_list)
    AVG_R = sum(R_list)/len(R_list)
    AVG_F1 = sum(F1_list)/len(F1_list)
    SCORE = 0.5 * (AVG_J + AVG_F1)
    return AVG_J, AVG_P, AVG_R, AVG_F1, SCORE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/data/lzm/DrugRecommend/src/worker/dataset/eval.jsonl", help="GT jsonl（每行一个样本）")
    ap.add_argument("--gt-id-field", default="就诊标识")
    ap.add_argument("--gt-label-field", default="出院带药列表")
    ap.add_argument("--pred-id-field", default="ID")
    ap.add_argument("--pred-label-field", default="prediction")
    ap.add_argument("--case-sensitive", action="store_true", help="区分大小写；默认不区分")
    args = ap.parse_args()
    pred_file = get_recommend(min_frequency=1, test_file=args.gt, save_path='/data/lzm/DrugRecommend/resource/output/val')
    lower = not args.case_sensitive
    gt = load_gt_jsonl(args.gt, args.gt_id_field, args.gt_label_field, lower)
    pred = load_pred(pred_file, args.pred_id_field, args.pred_label_field, lower)

    AVG_J, AVG_P, AVG_R, AVG_F1, SCORE = evaluate(gt, pred)
    print(f"AVG_Jaccard: {AVG_J:.6f}")
    print(f"AVG_Precision: {AVG_P:.6f}")
    print(f"AVG_Recall: {AVG_R:.6f}")
    print(f"AVG_F1: {AVG_F1:.6f}")
    print(f"Final Score = 0.5 * (Jaccard + F1) = {SCORE:.6f}")

if __name__ == "__main__":
    pred_file = get_recommend(min_frequency=1, test_file='/data/lzm/DrugRecommend/src/data/CDrugRed-B-v1/CDrugRed_test-B.jsonl', save_path ='/data/lzm/DrugRecommend/resource/output/submit')
    # main()