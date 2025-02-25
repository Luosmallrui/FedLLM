from datasets import load_from_disk

dataset_path = "/Users/luosmallrui/Downloads/wiki_lingua-chinese"  # 数据集路径
test_dataset = load_from_disk(dataset_path)
print(test_dataset)
for item in test_dataset:
    print(item)

