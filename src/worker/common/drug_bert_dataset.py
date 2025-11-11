# 基于Encoder模型下 关于数据的处理

# 我们原始所有字段信息拆分成两个部分 输入到两个Bert Model中 视为跨模态建模范式

# 第一个部分：仅包含诊疗过程描述字段，从最后面截断，从最后面填充

# 第二个部分：包含入院情况、现病史、既往史、主诉、出院诊断。从前面截断（我们认为出院诊断的信息会更重要一些），从后面填充。
import numpy as np
import json
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

from src.worker.common.data_process import load_drug_vocab, create_multihot_labels

class DrugTrainDataset(Dataset):
    def __init__(self, data_path: str = None, drug_file_path: str = None):
        self.drug_to_idx, self.idx_to_drug, self.drugs = load_drug_vocab(drug_file_path)
        self.mode_1_data, self.mode_2_data, self.labels = self.process_train_data(data_path)

    def safe_get_field(self, record, field_name, default=""):
        value = record.get(field_name)
        if value is None:
            return default
        if isinstance(value, list):
            return " ".join([str(x) for x in value])
        return str(value)
    def safe_get_field_in(self, record, field_name, default=""):
        value = record.get(field_name)
        if value is None:
            return default
        if isinstance(value, list):
            return "、".join([str(x) for x in value])
        return str(value)
    
    def process_train_data(self, data_path):
        mode_1_data = []
        mode_2_data = []
        labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                chief_complaint = self.safe_get_field(record, '主诉')
                diagnosis_desc = self.safe_get_field(record, '诊疗过程描述')
                hospital_admission_status = self.safe_get_field(record, '入院情况')
                present_illness = self.safe_get_field(record, '现病史')
                past_illness = self.safe_get_field(record, '既往史')
                discharge_diagnosis = self.safe_get_field_in(record, '出院诊断')
                mode_1_input = chief_complaint + " " + diagnosis_desc
                
                mode_2_input = (
                    hospital_admission_status + " " + 
                    present_illness + " " + 
                    past_illness + " " + 
                    chief_complaint + " " + 
                    "出院诊断：" + 
                    discharge_diagnosis
                )
                
                drug_list = record.get('出院带药列表', [])
                if not isinstance(drug_list, list):
                    drug_list = []
                if len(drug_list)==0:
                    print(f"发现标签长度为0的样本：{record.get('就诊标识')}")
                    continue
                mode_1_data.append(mode_1_input.strip())
                mode_2_data.append(mode_2_input.strip())
                labels.append(create_multihot_labels(drug_list, self.drug_to_idx, len(self.drugs)))
        
        return mode_1_data, mode_2_data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        mode_1 = self.mode_1_data[index]
        mode_2 = self.mode_2_data[index]
        label = self.labels[index]
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.float32)
        return {
            "mode_1": mode_1,
            "mode_2": mode_2,
            "labels": label
        }


def create_dataset(train_path, val_path, drug_file_path):
    train_dataset = DrugTrainDataset(data_path=train_path, drug_file_path=drug_file_path)
    val_dataset = DrugTrainDataset(data_path=val_path, drug_file_path=drug_file_path)
    return train_dataset, val_dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

if __name__ == '__main__':
    train_dataset, val_dataset = create_dataset("/data/lzm/DrugRecommend/src/worker/dataset/train.jsonl",
                                                 "/data/lzm/DrugRecommend/src/worker/dataset/eval.jsonl",
                                                   "/data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json")
    num_classes = len(train_dataset.drugs)
    samplers = [None, None]
    train_loader, val_loader = create_loader([train_dataset, val_dataset], samplers,
                                                          batch_size=[1] + [
                                                              1],
                                                          num_workers=[4, 4],
                                                          is_trains=[True, False],
                                                          collate_fns=[None, None])
    for i, batch in enumerate(train_loader):
        mode_1 = batch["mode_1"]
        mode_2 = batch["mode_2"] 
        labels = batch["labels"]
        print("train dataset")
        print("="*100)
        print("mode_1")
        print(mode_1)
        print("="*100)
        print("mode_2")
        print(mode_2)
        print("="*100)
        print(labels)
        print("="*100)
        break
    for i, batch in enumerate(val_loader):
        mode_1 = batch["mode_1"]
        mode_2 = batch["mode_2"] 
        labels = batch["labels"]
        print("val dataset")
        print("="*100)
        print("mode_1")
        print(mode_1)
        print("="*100)
        print("mode_2")
        print(mode_2)
        print("="*100)
        print(labels)
        print("="*100)
        break