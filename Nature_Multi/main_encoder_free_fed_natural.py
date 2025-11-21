import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import torch
import copy
from tqdm import tqdm
import numpy as np
import logging
import warnings
from datetime import datetime
import sys
import time
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(3000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from federated_learning import *
from config import get_config, get_training_config
from showomodel.vlm import init_vlm_model_lora,make_supervised_data_module_clients,set_selected_state_dict_lora,get_selected_state_dict_lora
from omegaconf import OmegaConf
from pathlib import Path
from pycocoevalcap.metrics_for_chs_mrg import compute_scores
import random
import glob

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    config.experiment.output_dir = f"{config.experiment.output_dir}/{config.dataset.dataset_name}_{config.dataset.split_strategy}_{config.fed.fed_alg}_L{config.optimizer.params.learning_rate}__b{config.training.batch_size_mmu}_l{config.dataset.preprocessing.max_seq_length}_{now_time}"

    return config

def save_config(config):
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    config_path = Path(config.experiment.output_dir) / "config.yaml"
    logging.info(f"Saving config to {config_path}")
    OmegaConf.save(config, config_path)

def setup_logging(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = os.path.join(logdir, 'training.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
    
    logging.info("logdir: %s", logfile)


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[0]

    all_labels = []
    all_preds = []
    for case_labels, case_preds in zip(labels, preds):
        case_lab,case_pred=[],[]
        for label, pred in zip(case_labels, case_preds):
            if label != -100:
                case_lab.append(label)
                case_pred.append(pred)
        all_labels.append(tokenizer.decode(case_lab))
        all_preds.append(tokenizer.decode(case_pred))

    if all_labels[0].find("caption")!=-1:
        gts={}
        res={}
        iid=0
    
        for label_sentence, pred_sentence in zip(all_labels, all_preds):          
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                gts[iid]=[label_sentence[label_is_pos + 4:-14]]
                res[iid]=[pred_sentence[pred_is_pos + 4:-14]]
                iid+=1
                if iid==10:
                    print(gts,res)
        result=compute_scores(gts, res, method="English_word", tokenizer=None, temp=True)
        print("len:",len(gts))
        return {
            "CIDEr": result["CIDEr"],
            "ROUGE_L": result["ROUGE_L"],
        }    


    elif all_labels[0].find("classification")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "class_accuracy": accuracy,
        }
    elif all_labels[0].find("question")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "vqa_accuracy": accuracy,
        }
    elif all_labels[0].find("coordinates")!=-1:
        print(all_preds[0:5])
        print("label format:",all_labels[0])
        iou_scores = []
        len_ok=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            label_is_pos = label_sentence.find("are")
            pred_is_pos = pred_sentence.find("are")
            if label_is_pos != -1 and pred_is_pos != -1:
                len_ok+=1
                label_pos = label_sentence[label_is_pos + 3:-14]
                answer_pos = pred_sentence[pred_is_pos + 3:-14]
                print("pre_str",answer_pos)
                print("pre format valid:",is_valid_box_format(answer_pos))
              
                if label_pos and answer_pos and is_valid_box_format(label_pos) and is_valid_box_format(answer_pos):
                    import ast
                    label_boxes = ast.literal_eval(label_pos)
                    try:
                        pred_boxes = ast.literal_eval(answer_pos)
                    except ValueError as e:
                        print(f"Error parsing answer_pos: {answer_pos}")
                        pred_boxes = None  

                    if pred_boxes != None and len(label_boxes) == 4 and len(pred_boxes) == 4:
                        label_boxes = [label_boxes]
                        pred_boxes = [pred_boxes]
                        for label_box, pred_box in zip(label_boxes, pred_boxes):
                            # print("label_boxes:",label_box,type(label_box))
                            # print("pred_boxes:",pred_box,type(pred_box))

                            if isinstance(label_box, dict) and isinstance(pred_box, dict):
                                for key in label_box:
                                    if key in pred_box:
                                        iou = compute_iou(label_box[key], pred_box[key])
                                        iou_scores.append(iou)
                                        if random.random() < 0.05:
                                            print("label_box:",label_box)
                                            print("pred_box:",pred_box)
                                            print("iou:",iou)
                                            print("=====================================")
                            elif isinstance(label_box, list) and isinstance(pred_box, list):
                                iou = compute_iou(label_box, pred_box)
                                iou_scores.append(iou)
                                if random.random() < 0.05:
                                    print("label_box:",label_box)
                                    print("pred_box:",pred_box)
                                    print("iou:",iou)
                                    print("=====================================")

        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        print("len:",len(iou_scores))
        print("len_ok:",len_ok)

        return {
            "mean_iou": mean_iou,
        }
    else:
        return {
                "mean_iou": 0,
            }

def compute_metrics_fed_med(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[0]

    all_labels = []
    all_preds = []
    for case_labels, case_preds in zip(labels, preds):
        case_lab, case_pred = [], []
        for label, pred in zip(case_labels, case_preds):
            if label != -100:
                case_lab.append(label)
                case_pred.append(pred)
        all_labels.append(tokenizer.decode(case_lab))
        all_preds.append(tokenizer.decode(case_pred))

    if all_labels[0].find("caption") != -1:
        gts = {}
        res = {}
        iid = 0

        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                gts[iid] = [label_sentence[label_is_pos + 4:-14]]
                res[iid] = [pred_sentence[pred_is_pos + 4:-14]]
                iid += 1
                if iid == 10:
                    print(gts, res)
        result = compute_scores(gts, res, method="English_word", tokenizer=None, temp=True)
        print("len:", len(gts))
        return {
            "CIDEr": result["CIDEr"],
            "ROUGE_L": result["ROUGE_L"],
        }


    elif all_labels[0].find("classification") != -1:
        right = 0
        all_num = 0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random() < 0.005:
                print("label:", label_sentence)
                print("pred:", pred_sentence)
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                la = label_sentence[label_is_pos + 3:label_is_pos + 4]
                answer = pred_sentence[pred_is_pos + 3:pred_is_pos + 4]
                if la == answer and la != " ":
                    right += 1
            all_num += 1

        accuracy = right / all_num
        print(right, all_num, accuracy)

        return {
            "class_accuracy": accuracy,
        }
    elif all_labels[0].find("question") != -1:
        right = 0
        all_num = 0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random() < 0.005:
                print("label:", label_sentence)
                print("pred:", pred_sentence)
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                la = label_sentence[label_is_pos + 3:label_is_pos + 4]
                answer = pred_sentence[pred_is_pos + 3:pred_is_pos + 4]
                if la == answer and la != " ":
                    right += 1
            all_num += 1

        accuracy = right / all_num
        print(right, all_num, accuracy)

        return {
            "vqa_accuracy": accuracy,
        }
    elif all_labels[0].find("object") != -1 or all_labels[0].find("objects") != -1:
        # 打印前5个预测结果和标签格式进行调试
        print(all_preds[0:5])
        print("label format:", all_labels[0])

        iou_scores = []
        len_ok = 0

        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            label_pos_str=""
            answer_pos_str=""

            # 1. 寻找坐标的起始和结束位置
            # 假设坐标数据总是在 '{}' 或 '[]' 中
            label_start = label_sentence.find('is')
            if label_start == -1:  # 如果没有找到 'object is '，则尝试找 'objects are '
                label_start = label_sentence.find('are')
                label_end= label_sentence.find('].')
                if label_start != -1 and label_end != -1:
                    label_pos_str = label_sentence[label_start+3: label_end + 1]
            else:
                label_end = label_sentence.find('}.')
                if label_start != -1 and label_end != -1:
                    label_pos_str = label_sentence[label_start+2: label_end + 1]

            pred_start = pred_sentence.find('is')
            if pred_start == -1:  # 如果没有找到 'object is '，则尝试找 'objects are '
                pred_start = pred_sentence.find('are')
                pred_end = pred_sentence.find('].')
                if pred_start != -1 and pred_end != -1:
                    answer_pos_str = pred_sentence[pred_start+3: pred_end + 1]
            else:
                pred_end = pred_sentence.find('}.')
                if pred_start != -1 and pred_end != -1:
                    answer_pos_str = pred_sentence[pred_start+2: pred_end + 1]

            # 2. 提取坐标字符串
            if label_pos_str and answer_pos_str:
                len_ok += 1
                print("label_pos_str",label_pos_str)
                print("pre_str:", answer_pos_str) # 调试输出

                # 3. 解析字符串为 Python 对象
                import ast

                # 初始化解析结果
                pred_objects = []

                # 使用 ast.literal_eval 安全地解析
                try:
                    parsed_label = ast.literal_eval(label_pos_str)
                    parsed_answer = ast.literal_eval(answer_pos_str)
                except ValueError as e:
                    # 预测结果解析失败，记录错误并跳过
                    print(f"Error parsing pred_sentence: {answer_pos_str}. Error: {e}")
                    continue

                # 4. 统一化处理格式：将单个对象转换为列表
                # 标签格式可能是 {'heart': [...] } (dict) 或 [{'lung': [...] }, {...}] (list)
                if isinstance(parsed_label, dict):
                    label_objects = list(parsed_label.items())  # 转换为 [(key, box)] 格式
                elif isinstance(parsed_label, list):
                    label_objects = parsed_label
                else:
                    continue

                if isinstance(parsed_answer, dict):
                    pred_objects = list(parsed_answer.items())
                elif isinstance(parsed_answer, list):
                    pred_objects = parsed_answer

                # 5. 迭代并计算 IoU
                # 注意：对于多对象，我们采用预测结果中的所有框与标签中的所有框进行比较

                # 为简化逻辑，我们要求预测和标签的框数量必须一致（或采用最佳匹配策略）
                # 这里使用简化策略：数量必须一致，且假设顺序相同（如果标签是列表）

                if len(label_objects) != len(pred_objects):
                    # 如果数量不匹配，则跳过或采用更复杂的匹配算法
                    continue

                for label_item, pred_item in zip(label_objects, pred_objects):

                    # 提取坐标列表
                    if isinstance(label_item, tuple):  # 标签是 (key, box_list)
                        label_box = label_item[1]
                        pred_box = pred_item[1] if isinstance(pred_item, tuple) else list(pred_item.values())[0]
                    elif isinstance(label_item, dict):  # 标签是 {'key': box_list}
                        label_box = list(label_item.values())[0]
                        pred_box = list(pred_item.values())[0]
                    else:
                        continue  # 格式不符合预期

                    if len(label_box) == 4 and len(pred_box) == 4 and is_valid_box_format(label_box) and is_valid_box_format(pred_box):
                        # 计算 IoU
                        iou = compute_iou(label_box, pred_box)
                        iou_scores.append(iou)

                        # 随机打印调试信息
                        if random.random() < 0.05:
                            print("Label Box:", label_box)
                            print("Pred Box:", pred_box)
                            print("IoU:", iou)
                            print("=====================================")



        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        print("len:", len(iou_scores))
        print("len_ok:", len_ok)

        return {
            "mean_iou": mean_iou,
        }
    else:
        return {
            "mean_iou": 0,
        }


config = get_config()

output_dir = config.experiment.output_dir
setup_logging(output_dir)
save_config(config)
set_seed(config.training.seed)

writer = SummaryWriter(log_dir=output_dir+'/vis')

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
logging.info("===============================================================")
logging.info("===============================================================")
logging.info("===============================================================")


logging.info("test_local: %s",config.fed.test_local)
logging.info("batch_size: %d",config.training.batch_size_mmu) 


try:
    ## ===== laod model =====
    model,tokenizer=init_vlm_model_lora(config)
    # ===== Load the dataset =====
    client_files = glob.glob(config.dataset.data_path + "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    config.fed.num_clients = len(client_files)
    config.fed.sample_clients = len(client_files)


    local_datasets, global_test_datasets, sub_test_datasets, data_collator = make_supervised_data_module_clients(tokenizer, config) 


    num_clients=config.fed.num_clients
    output_dir = config.experiment.output_dir

    sample_num_list = [len(local_datasets[i]) for i in range(num_clients)]
    logging.info("sample_num_list: %s",sample_num_list) 


    # ===== Define the global and local models =====
    share_params= get_selected_state_dict_lora(model)

    global_dict = copy.deepcopy(share_params)
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(num_clients)]
    
    if config.fed.fed_alg in ['fedprox','fedadagrad', 'fedyogi', 'fedadam',"fedavgm"]:
        for key in global_dict.keys():
            global_dict[key] = global_dict[key].to(device)

    proxy_dict, opt_proxy_dict = get_proxy_dict(config.fed, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(config.fed, global_dict)
    # if config.fed.fed_alg in ['scaffold']:
    #     for key in global_auxiliary.keys():
    #         global_auxiliary[key] = global_auxiliary[key].to(device)
    #     auxiliary_model_list = auxiliary_model_list.to(device)
    #     for key in auxiliary_delta_dict.keys():
    #         auxiliary_delta_dict[key] = auxiliary_delta_dict[key].to(device)

    # ===== Start federated training =====
    training_loss = [[] for _ in range(num_clients)]
    test_metrics = [{} for _ in range(num_clients)]

    select_keys=["eval_class_accuracy","eval_vqa_accuracy","eval_loss", "eval_mean_iou","eval_CIDEr","eval_ROUGE_L"]
    # if config.dataset.dataset_name=='Fed-Med':
    #     compute_metrics=compute_metrics_fed_med
    # elif config.dataset.dataset_name=='Fed-Nature':
    #     compute_metrics=compute_metrics_fed_nature

    for round in tqdm(range(config.fed.num_rounds)):
        t_start=time.time()
        clients_this_round = get_clients_this_round(config.fed, round)

        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
        for client in range(num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue
            logging.info(f">> ============: {client} ==============")
            set_selected_state_dict_lora(model, global_dict)  
 
            new_lr = cosine_learning_rate(round, config.fed.num_rounds, config.optimizer.params.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_config(config, new_lr)
            logging.info(f"round: {round}, new_lr: {new_lr}")
            logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

            # checkpoint_path=os.path.join(output_dir, f"checkpoint-{round}") if round>0 else None
            # ===== Train local model on the client side =====
            trainer = get_fed_showo_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=local_datasets[client],
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                global_dict=global_dict,
                config=config,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                eval_dataset=None,
            )
            logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")
            results = trainer.train()
            # --- 关键修改：在这里添加显式删除 ---

            # 1. 显式删除 Trainer 内部的 Optimizer 和 Scheduler
            #    这是释放显存的关键一步，因为它们持有庞大的状态缓冲区
            if client<num_clients-1:
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    print("释放optimizer显存")
                    del trainer.optimizer
                    logging.info(
                        f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")


                if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
                    print("释放scheduler显存")
                    del trainer.lr_scheduler
                    logging.info(
                        f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")

                if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
                    print("释放tokenizer显存")
                    del trainer.tokenizer
                    logging.info(
                        f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")

                model.cpu()
                # 2. 删除 Trainer 对象
                del trainer

                # 3. 清理 CUDA 缓存
                #    现在 Optimizer 已经被删除，empty_cache() 就能回收它占用的显存
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")
            training_loss[client].append(results.training_loss)
            logging.info(results)

            # ===== Client transmits local information to server =====
            if config.fed.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            local_dict_list[client] = copy.deepcopy(get_selected_state_dict_lora(model))   # copy is needed!
            # upload_params = model.model.state_dict()
            # local_dict_list[client] = copy.deepcopy(upload_params)
            # for key in local_dict_list[client]:
            #     local_dict_list[client][key] = local_dict_list[client][key].cpu()


            # if config.fed.test_local and round % 3 == 0 and round >=0 :
            #     print("local eval")
            #     for idx,subdata in enumerate(sub_test_datasets):
            #         eval_results = trainer.evaluate(eval_dataset=subdata)
            #         with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
            #             f.write(f"    ---------------- Clinet{client} : subdataset{idx}, {eval_results}\n")
            #     eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
            #     with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
            #         f.write(f"     - Round {round+1} Clinet{client} : {eval_results}\n")
            #         f.write("================================\n")
                                
            #     for key,value in eval_results.items():
            #         if key in select_keys:
            #             test_metrics[client][key] = eval_results[key]

     
        # ===== summary loss =====
        avg_loss = np.mean([training_loss[i][-1] for i in range(num_clients) if i in clients_this_round])
        writer.add_scalar('Train/Average Training Loss', avg_loss, round + 1)
        if config.fed.test_local and round % 3 == 0 and round > 5:
            for key in select_keys:
                key_values = [test_metrics[client][key] for client in range(num_clients) if key in test_metrics[client]]
                if key_values:
                    avg_key, std_key = compute_avg_and_std(key_values)
                    logging.info(f"Round {round+1}, Key: {key}, Avg: {avg_key}, Std: {std_key}")
                    writer.add_scalar(f"local_test_{key}/Avg", avg_key, round + 1)
                    writer.add_scalar(f"local_test_{key}/Std", std_key, round + 1)
                    with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                        f.write(f"    summary - Round {round+1} : {key}, Avg: {avg_key}, Std: {std_key}\n")
  
        # ===== Server aggregates the local models =====
        use_cpu=True
        if num_clients>5:
            use_cpu=True
        if config.fed.fed_alg == 'scaffold':
            for key in global_auxiliary.keys():
                global_auxiliary[key] = global_auxiliary[key].to(device)

        global_dict, global_auxiliary = global_aggregate(
            config.fed, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),use_cpu=use_cpu
        )
        set_selected_state_dict_lora(model, global_dict)  
        torch.cuda.empty_cache()  

        # ===== Test the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

        for area_dataset in global_test_datasets:
          
            eval_results = trainer.evaluate(eval_dataset=area_dataset)
            logging.info(area_dataset.dataset_name ,eval_results)
            for key in select_keys:
                if key in eval_results:
                    writer.add_scalar(f"Global_test/Average {key}", eval_results[key], round + 1)
            with open(os.path.join(output_dir, "eval_results.txt"), "a") as f:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if "eval_vqa_accuracy" in eval_results:
                    task_name = "VQA"
                elif "eval_class_accuracy" in eval_results:
                    task_name = "classification"
                elif "eval_mean_iou" in eval_results:
                    task_name = "Detection"
                else:
                    task_name = "Caption"
                f.write(f"{current_time} - Round {round+1}: {task_name} {eval_results}\n")



        if (round+1) % config.experiment.save_model_freq == 0:
            logging.info(f"checkpoint-{round+1}")
            trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
        t_end=time.time()
        logging.info(f"round time: {(t_end-t_start)/3600:.2f} hours")

    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, \n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, \n")
        f.write("======================================================\n")
    raise e