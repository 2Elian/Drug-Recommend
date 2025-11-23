# tests
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
def check_neo4j_labels():
    """检查Neo4j中的实际标签名称"""
    NEO4J_URI = "bolt://172.16.107.15:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "MyStrongPassword123"
    kg = MedicalKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # 查询所有节点标签
        query = "CALL db.labels()"
        with kg.driver.session() as session:
            results = session.run(query)
            labels = [record['label'] for record in results]
            print("Neo4j中的标签:", labels)
        
        # 查询所有关系类型
        query = "CALL db.relationshipTypes()"
        with kg.driver.session() as session:
            results = session.run(query)
            relationships = [record['relationshipType'] for record in results]
            print("Neo4j中的关系类型:", relationships)
            
    except Exception as e:
        print(f"检查Neo4j元数据失败: {e}")
    finally:
        kg.close()

# 运行检查
check_neo4j_labels()