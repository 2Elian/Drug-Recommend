import json
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any

class MedicalKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空")
    
    def create_constraints(self):
        """创建约束确保唯一性"""
        with self.driver.session() as session:
            try:
                # 为疾病创建唯一约束
                session.run("CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
                # 为药物创建唯一约束
                session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE")
                # 为患者创建唯一约束
                session.run("CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE")
                print("约束创建完成")
            except Exception as e:
                print(f"创建约束时出现警告: {e}")
    
    def create_patient_diagnosis_drugs(self, patient_data: Dict[str, Any]):
        """创建患者-诊断-药物的关系"""
        with self.driver.session() as session:
            # 新版本使用 execute_write 而不是 write_transaction
            result = session.execute_write(
                self._create_patient_diagnosis_drugs_tx, patient_data
            )
            return result
    
    @staticmethod
    def _create_patient_diagnosis_drugs_tx(tx, data):
        """事务函数：创建患者-诊断-药物关系"""
        
        # 创建患者节点
        patient_id = data.get("patient_id", f"patient_{hash(str(data))}")
        tx.run(
            "MERGE (p:Patient {id: $patient_id})",
            patient_id=patient_id
        )
        
        # 处理诊断和药物 - 过滤空值
        diagnoses = [d for d in data.get("出院诊断", []) if d and d.strip()]
        drugs = [d for d in data.get("出院带药列表", []) if d and d.strip()]
        
        # 创建诊断节点并建立关系
        for diagnosis in diagnoses:
            # 创建疾病节点
            tx.run(
                "MERGE (d:Disease {name: $diagnosis})",
                diagnosis=diagnosis
            )
            # 创建患者-诊断关系
            tx.run(
                """
                MATCH (p:Patient {id: $patient_id}), (d:Disease {name: $diagnosis})
                MERGE (p)-[r:DIAGNOSED_WITH]->(d)
                SET r.timestamp = timestamp()
                """,
                patient_id=patient_id,
                diagnosis=diagnosis
            )
        
        # 创建药物节点并建立关系
        for drug in drugs:
            # 创建药物节点
            tx.run(
                "MERGE (dr:Drug {name: $drug})",
                drug=drug
            )
            # 创建患者-药物关系（出院带药）
            tx.run(
                """
                MATCH (p:Patient {id: $patient_id}), (dr:Drug {name: $drug})
                MERGE (p)-[r:PRESCRIBED]->(dr)
                SET r.type = '出院带药', r.timestamp = timestamp()
                """,
                patient_id=patient_id,
                drug=drug
            )
        
        # 创建诊断-药物关系（基于同一患者的关联）
        for diagnosis in diagnoses:
            for drug in drugs:
                tx.run(
                    """
                    MATCH (d:Disease {name: $diagnosis}), (dr:Drug {name: $drug})
                    MERGE (d)-[r:TREATED_WITH]->(dr)
                    SET r.weight = coalesce(r.weight, 0) + 1
                    """,
                    diagnosis=diagnosis,
                    drug=drug
                )
        
        return len(diagnoses), len(drugs)

def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"加载数据失败: {e}")
    return data

def process_data_for_kg(raw_data: List[Dict]) -> List[Dict]:
    """处理原始数据，添加患者ID等必要字段"""
    processed_data = []
    empty_diagnoses_count = 0
    empty_drugs_count = 0
    
    for i, record in enumerate(raw_data):
        # 过滤空值
        diagnoses = [d for d in record.get("出院诊断", []) if d and d.strip()]
        drugs = [d for d in record.get("出院带药列表", []) if d and d.strip()]
        
        if not diagnoses:
            empty_diagnoses_count += 1
        if not drugs:
            empty_drugs_count += 1
            
        processed_record = {
            "patient_id": f"patient_{i+1:04d}",
            "出院诊断": diagnoses,
            "出院带药列表": drugs
        }
        processed_data.append(processed_record)
    
    print(f"数据统计:")
    print(f"  - 总记录数: {len(processed_data)}")
    print(f"  - 空诊断记录: {empty_diagnoses_count}")
    print(f"  - 空药物记录: {empty_drugs_count}")
    
    return processed_data

# 主执行函数 - 只构建知识图谱
def build_knowledge_graph():
    # Neo4j连接配置
    NEO4J_URI = "bolt://172.16.107.15:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "MyStrongPassword123"
    
    try:
        # 初始化知识图谱
        print("正在初始化 Neo4j 连接...")
        kg = MedicalKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # 测试连接
        kg.driver.verify_connectivity()
        print("Neo4j 连接成功！")
        
        # 清空数据库
        print("清空数据库...")
        kg.clear_database()
        
        # 创建约束
        print("创建约束...")
        kg.create_constraints()
        
        # 加载数据
        print("加载数据...")
        raw_data = load_jsonl_data("/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl")
        
        if not raw_data:
            print("没有加载到数据，退出程序")
            return
        
        processed_data = process_data_for_kg(raw_data)
        
        # 构建知识图谱
        total_diagnoses = 0
        total_drugs = 0
        
        print(f"开始构建知识图谱，共 {len(processed_data)} 条记录...")
        for i, record in enumerate(processed_data):
            try:
                diagnoses_count, drugs_count = kg.create_patient_diagnosis_drugs(record)
                total_diagnoses += diagnoses_count
                total_drugs += drugs_count
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1} 条记录")
            except Exception as e:
                print(f"处理第 {i+1} 条记录时出错: {e}")
                continue
        
        print(f"知识图谱构建完成！")
        print(f"总共处理: {len(processed_data)} 个患者")
        print(f"诊断数量: {total_diagnoses}")
        print(f"药物数量: {total_drugs}")
        
    except Exception as e:
        print(f"构建过程中出现错误: {e}")
    finally:
        kg.close()

if __name__ == "__main__":
    build_knowledge_graph()