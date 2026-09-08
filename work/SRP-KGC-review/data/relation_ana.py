import json

# 读取JSON文件
with open('relations.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 写入到TXT文件
with open('relation2id.txt', 'w', encoding='utf-8') as txt_file:
    for index, (key, value) in enumerate(data.items()):
        txt_file.write(f"{key}  {index}\n")
with open('entities.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取entity_id并标号
entity_ids = []
for index, item in enumerate(data):
    entity_id = item.get('entity_id')
    if entity_id:
        entity_ids.append(f"{entity_id} {index}")

# 将结果写入entity2id.txt文件
with open('entity2id.txt', 'w', encoding='utf-8') as file:
    for entity in entity_ids:
        file.write(entity + '\n')