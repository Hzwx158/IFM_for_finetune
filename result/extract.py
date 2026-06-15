import yaml
import json
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
from argparse import ArgumentParser

RES_DIR = Path(__file__).parent.absolute()
assert RES_DIR.exists(), f"RES_DIR: <{RES_DIR}> not exist"

DATASETS = (
    "cifar100",
    "svhn",
    "food101",
)

WAY_NAME={
    "ffn_blr_0.01": "AdaptFormer",
    "merge_before_tune_0.99_blr_0.01": "IFM before finetune (Ours)",
    "lora_blr_0.01": "LORA",
    
    "fulltune_blr_0.01": "Full fine-tune",
    "fulltune_value_only_include_v_blr_0.01": "Tune Value Only",
    
    
    "merge_before_tune_0.8_blr_0.01": "IFM before finetune (Ours)",
    "expand_blr_only_new_0.01": "Expand",
}

def get_dir_name(ds_name:str, way_name: str): 
    return f"{ds_name}_{way_name}"

def extract_from_log(log_file:Path) -> Optional[dict[int, dict]]:
    assert log_file.exists()
    assert log_file.is_file()
    assert log_file.suffix == ".txt"
    with open(log_file, "r") as f:
        records = list(map(json.loads, f.readlines()))
    
    res = {}
    
    # split
    seed_indices = [
        idx
        for idx, record in enumerate(records)
        if 'seed' in record
    ]
    # read
    for seed_idx in reversed(seed_indices): # 倒序保证用最新的
        
        seed_record = records[seed_idx]
        seed = seed_record['seed']
        if seed in res:
            continue
        
        begin_idx = seed_idx - 100
        assert begin_idx >= 0
        
        max_acc_info = res[seed] = seed_record
        max_acc_info.update(records[begin_idx])
        for record in records[begin_idx+1:seed_idx]:
            if record['test_acc1'] >= max_acc_info['test_acc1']:
                max_acc_info.update(record)
    
    # if res.keys() != {42, 43, 44, 45, 46}:
    #     return None

    return res if res else None

ResultsT = dict[str, dict[str, dict[str, float|str|int]]]

def extract_all_as_dict() -> ResultsT:
    
    res:ResultsT = {}
    '''
    res = dict[
        model-name:str,
        model-res: dict[
            way-name:str,
            way-metric: dict[name:str, item:float|str|int]
        ]
    ]
    '''
    for way_path in RES_DIR.iterdir():
        if not way_path.is_dir(): continue
        way_name = way_path.name
        print(f"way: {way_name}")
        for model_res_path in way_path.iterdir():

            seed_result_map = extract_from_log(model_res_path / "log.txt")
            if seed_result_map is None:
                continue
            
            model_name = model_res_path.name
            print(f"model: {model_name}")
            model_name, tag = model_name.split("_converted")
            if model_name not in res:
                res[model_name] = {}
            model_res = res[model_name]
            
            test_accs = []            
            for seed, result in seed_result_map.items():
                # print(result)
                test_accs.append(result['test_acc1'])
            test_accs = np.array(test_accs)
            
            
            model_res[way_name] = dict(
                acc_mean=np.mean(test_accs).item(),
                acc_std=np.std(test_accs).item(),
                tag=tag,
                seeds=', '.join(map(str, seed_result_map.keys())),
                # n_param=result['n_parameters'],
            )
    return res

def make_acc_str(d:dict, k:str) -> str | None:
    if k not in d:
        return None
    dd = d[k]
    acc_mean = dd['acc_mean']
    acc_std = dd['acc_std']
    
    return f"${acc_mean:.2f} \\pm {acc_std:.2f}\\%$"
    

def generate_latex_items(
    data:ResultsT, 
    ways:list[str], 
    only_resnet:bool,
    show_trainable_args = False
) -> None:
    
    res = []
    
    for model_name, model_info in data.items():
        if only_resnet:
            if 'resnet' not in model_name: 
                continue
        else:
            if 'resnet' in model_name: 
                continue
        if not model_info:
            continue
        
        model_name_str = "\multirow{5}{*}{\makecell[c]{" + model_name.replace('_', '-\\\\') + "}}"
        
        res += [
            '\\hline',
            model_name_str,
        ]
        
        for way in ways:
            acc_list = [
                make_acc_str(model_info, f"{ds}_{way}") or "?"
                for ds in DATASETS
            ]
            if show_trainable_args:
                with open(
                    RES_DIR / f"food101_{way}" / 
                    (f"{model_name}_converted"+model_info[f"food101_{way}"]['tag']) / 
                    "trainable_args.txt", 'r'
                ) as f:
                    lines = f.readlines()[1:-1]
                    kvs = {
                        k: v.replace(',', '')
                        for l in lines 
                        for k,v in (l.strip().split(': '), )
                    }
                    trainable_args = format(float(kvs["Trainable Parameters"])/1_000_000, '.2f')
                    trainable_ratio = kvs["Trainable Ratio"][:-3]+"\\%"
            else:
                trainable_args = '?'
                trainable_ratio = '?'

            # & Full fine-tune    &  86.04M ($100\%$) &  $86.95\%$ & $97.61\%$ & $89.94\%$\\
            s = (
                f"& {WAY_NAME[way]} & "+
                f"{trainable_args}M ({trainable_ratio}) & " + 
                ' & '.join(acc_list) + 
                "\\\\"
            )
            
            res.append(s)
            if 'value_only' in way:
                res.append("\\cline{2-6}")
            
        res.append('')
        
    final_str = '\n'.join(res)
    
    with open(RES_DIR/"tmp.txt", "w") as f:
        f.write(final_str)
    print(f'write into {RES_DIR/"tmp.txt"}')

def main():
    # args
    parser = ArgumentParser()
    parser.add_argument('--yaml', action='store_true', 
                        help="Save info to yaml file")
    parser.add_argument('--latex', action='store_true',
                        help="Change results into latex chart")
    args = parser.parse_args()
    # extract
    results = extract_all_as_dict()
    # yaml
    if args.yaml:
        with open(RES_DIR/'result.yaml', 'w') as f:
            yaml.safe_dump(results, f)
    # latex
    if args.latex:
        generate_latex_items(
            data = results,
            ways = [
                "fulltune_blr_0.01",
                "fulltune_value_only_include_v_blr_0.01",
                "ffn_blr_0.01", 
                "lora_blr_0.01",
                "merge_before_tune_0.99_blr_0.01",
                
                # "merge_before_tune_0.99_blr_0.01",
                # "expand_blr_only_new_0.01",
                # "fulltune_blr_0.01",
                # "parallel",
            ],
            only_resnet=False,
            show_trainable_args=False,
        )

if __name__ == '__main__':
    main()
    
    
    