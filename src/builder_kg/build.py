from typing import Dict, List, Tuple, Set
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Any
import json
from tqdm import tqdm
from neo4j import GraphDatabase
import re
from datetime import datetime

class MedicalKnowledgeGraph:
    # to neo4j
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, parameters: Dict = None):
        with self.driver.session() as session:
            return session.run(query, parameters)
    
    def create_constraints(self):
        """创建唯一性约束确保数据一致性"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Examination) REQUIRE e.name IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.execute_query(constraint)
            except Exception as e:
                print(f"创建约束失败: {e}")

class DrugKgBuilder:
    def __init__(self, file_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 ner_model_path: str):
        self.datas = self.load_jsonl_data(file_path)
        self.ner = ner_model_path
        self.kg = MedicalKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        self.kg.create_constraints()
        self.entity_cache = {
            'diseases': set(),
            'drugs': set(),
            'symptoms': set(),
            'examinations': set(),
            'patients': set()
        }
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f"加载数据失败: {e}")
        return data
    
    def extract_basic_entities(self, data: Dict) -> Dict[str, Any]:
        """提取基础实体"""
        basic_entities = {}
        patient_id = data.get('就诊标识', f"patient_{hash(str(data))}")
        basic_entities['patient'] = {
            'patient_id': patient_id,
            'gender': data.get('性别'),
            'birth_date': data.get('出生日期'),
            'ethnicity': data.get('民族'),
            'bmi': data.get('BMI'),
            'visit_time': data.get('就诊时间')
        }
        if data.get('出生日期'):
            birth_year = int(data['出生日期'].split('-')[0])
            visit_year = int(data['就诊时间'].split('-')[0]) if data.get('就诊时间') else datetime.now().year
            basic_entities['patient']['age'] = visit_year - birth_year
        if data.get('BMI'):
            bmi = data['BMI']
            if bmi < 18.5:
                basic_entities['patient']['bmi_category'] = '偏瘦'
            elif 18.5 <= bmi < 24:
                basic_entities['patient']['bmi_category'] = '正常'
            elif 24 <= bmi < 28:
                basic_entities['patient']['bmi_category'] = '偏胖'
            else:
                basic_entities['patient']['bmi_category'] = '肥胖'
        
        return basic_entities
    
    def extract_text_entities(self, text_fields: Dict[str, str]) -> Dict[str, List[str]]:
        # TODO 换成NER模型
        entities = {
            'diseases': set(),
            'symptoms': set(),
            'drugs': set(),
            'examinations': set(),
            'treatments': set()
        }
        
        # 诊疗过程
        disease_patterns = [
            r'([\u4e00-\u9fa5]+糖尿病)',
            r'([\u4e00-\u9fa5]+感染)',
            r'([\u4e00-\u9fa5]+病变)',
            r'([\u4e00-\u9fa5]+症)',
            r'([\u4e00-\u9fa5]+病)'
        ]
        
        # 入院情况
        symptom_patterns = [
            r'(烦渴|多饮|多尿|尿痛|头痛|发热|咳嗽)'
        ]
        
        # 现病史
        drug_patterns = [
            r'([\u4e00-\u9fa5]+胍)',
            r'([\u4e00-\u9fa5]+胰岛素)',
            r'([\u4e00-\u9fa5]+片)',
            r'([\u4e00-\u9fa5]+胶囊)'
        ]
        
        # 既往史
        exam_patterns = [
            r'([\u4e00-\u9fa5]+检查)',
            r'([\u4e00-\u9fa5]+试验)',
            r'([\u4e00-\u9fa5]+检测)'
        ]
        # 主诉
        
        all_text = " ".join(text_fields.values())
        
        # 提取疾病
        for pattern in disease_patterns:
            matches = re.findall(pattern, all_text)
            entities['diseases'].update(matches)
        
        # 提取症状
        for pattern in symptom_patterns:
            matches = re.findall(pattern, all_text)
            entities['symptoms'].update(matches)
        
        # 提取药物
        for pattern in drug_patterns:
            matches = re.findall(pattern, all_text)
            entities['drugs'].update(matches)
        
        # 提取检查
        for pattern in exam_patterns:
            matches = re.findall(pattern, all_text)
            entities['examinations'].update(matches)
        
        # 转换为列表
        return {k: list(v) for k, v in entities.items() if v}
    
    def extract_relations_with_llm(self, entities: Dict, text: str) -> List[Tuple]:
        # TODO 异步llm调用 + 提示词管理器
        relations = []
        
        # 模拟LLM提取的关系
        text_lower = text.lower()
        
        # 疾病-症状关系
        for disease in entities.get('diseases', []):
            for symptom in entities.get('symptoms', []):
                if disease in text and symptom in text:
                    # 简单的共现判断
                    relations.append(("疾病-症状关系", disease, symptom, f"{disease}表现为{symptom}"))
        
        # 疾病-药物关系
        for disease in entities.get('diseases', []):
            for drug in entities.get('drugs', []):
                if disease in text and drug in text:
                    relations.append(("疾病-药物关系", disease, drug, f"{disease}使用{drug}治疗"))
        
        # 症状-检查关系
        for symptom in entities.get('symptoms', []):
            for exam in entities.get('examinations', []):
                if symptom in text and exam in text:
                    relations.append(("症状-检查关系", symptom, exam, f"{symptom}通过{exam}确认"))
        
        return relations
    
    def build_patient_graph(self, data: Dict):
        """为单个患者构建知识图谱"""
        try:
            basic_entities = self.extract_basic_entities(data)
            patient_info = basic_entities['patient']
            text_fields = {
                '诊疗过程描述': data.get('诊疗过程描述', ''),
                '入院情况': data.get('入院情况', ''),
                '现病史': data.get('现病史', ''),
                '既往史': data.get('既往史', ''),
                '主诉': data.get('主诉', '')
            }
            text_entities = self.extract_text_entities(text_fields)
            diagnoses = data.get('出院诊断', [])
            if diagnoses:
                text_entities['diseases'] = list(set(text_entities.get('diseases', []) + diagnoses))
            combined_text = " ".join(text_fields.values())
            relations = self.extract_relations_with_llm(text_entities, combined_text)
            self._write_to_neo4j(patient_info, text_entities, relations, data.get('出院带药', []))
            
        except Exception as e:
            print(f"构建患者图谱失败: {e}")
    
    def _write_to_neo4j(self, patient_info: Dict, entities: Dict, relations: List, medications: List):
        patient_query = """
        MERGE (p:Patient {patient_id: $patient_id})
        SET p.gender = $gender,
            p.birth_date = $birth_date, 
            p.ethnicity = $ethnicity,
            p.bmi = $bmi,
            p.visit_time = $visit_time,
            p.age = $age,
            p.bmi_category = $bmi_category
        """
        self.kg.execute_query(patient_query, patient_info)

        for disease in entities.get('diseases', []):
            disease_query = """
            MERGE (d:Disease {name: $disease_name})
            MERGE (p:Patient {patient_id: $patient_id})
            MERGE (p)-[r:DIAGNOSED_WITH]->(d)
            SET r.diagnosis_time = $visit_time
            """
            self.kg.execute_query(disease_query, {
                'disease_name': disease,
                'patient_id': patient_info['patient_id'],
                'visit_time': patient_info.get('visit_time')
            })

        for symptom in entities.get('symptoms', []):
            symptom_query = """
            MERGE (s:Symptom {name: $symptom_name})
            MERGE (p:Patient {patient_id: $patient_id})
            MERGE (p)-[r:EXPERIENCES]->(s)
            """
            self.kg.execute_query(symptom_query, {
                'symptom_name': symptom,
                'patient_id': patient_info['patient_id']
            })
        
        for exam in entities.get('examinations', []):
            exam_query = """
            MERGE (e:Examination {name: $exam_name})
            MERGE (p:Patient {patient_id: $patient_id})
            MERGE (p)-[r:UNDERGOES]->(e)
            """
            self.kg.execute_query(exam_query, {
                'exam_name': exam,
                'patient_id': patient_info['patient_id']
            })
        
        for drug in medications:
            drug_query = """
            MERGE (dr:Drug {name: $drug_name})
            MERGE (p:Patient {patient_id: $patient_id})
            MERGE (p)-[r:PRESCRIBED]->(dr)
            SET r.prescription_type = '出院带药'
            """
            self.kg.execute_query(drug_query, {
                'drug_name': drug,
                'patient_id': patient_info['patient_id']
            })
        
        for rel_type, source, target, description in relations:
            if rel_type == "疾病-症状关系":
                rel_query = """
                MERGE (d:Disease {name: $source})
                MERGE (s:Symptom {name: $target})
                MERGE (d)-[r:HAS_SYMPTOM]->(s)
                SET r.description = $description,
                    r.source = '文本提取'
                """
            elif rel_type == "疾病-药物关系":
                rel_query = """
                MERGE (d:Disease {name: $source})
                MERGE (dr:Drug {name: $target})
                MERGE (d)-[r:TREATED_WITH]->(dr)
                SET r.description = $description,
                    r.source = '文本提取',
                    r.confidence = 0.7
                """
            elif rel_type == "症状-检查关系":
                rel_query = """
                MERGE (s:Symptom {name: $source})
                MERGE (e:Examination {name: $target})
                MERGE (s)-[r:CONFIRMED_BY]->(e)
                SET r.description = $description,
                    r.source = '文本提取'
                """
            else:
                continue
            
            self.kg.execute_query(rel_query, {
                'source': source,
                'target': target,
                'description': description
            })
    
    def build(self):
        print("开始构建知识图谱...")
        for data in tqdm(self.datas, desc='处理患者数据'):
            self.build_patient_graph(data)
        print("知识图谱构建完成！")
    
    def close(self):
        """关闭连接"""
        self.kg.close()

if __name__ == '__main__':
    drug_kg = DrugKgBuilder()
    drug_kg.build()
    drug_kg.close()