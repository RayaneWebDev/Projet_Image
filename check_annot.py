import os, json
ann_dir = 'annotation'
img_dir = 'data/validation'
found, missing = [], []
for jf in sorted(os.listdir(ann_dir)):
    if not jf.endswith('.json'):
        continue
    with open(os.path.join(ann_dir, jf)) as f:
        data = json.load(f)
    raw = data.get('imagePath', '')
    ip = raw.replace('\\', '/').split('/')[-1]
    img_path = os.path.join(img_dir, ip)
    if os.path.exists(img_path):
        found.append(jf)
    else:
        missing.append((jf, ip))
print(f'Found: {len(found)}, Missing: {len(missing)}')
for jf, img in missing[:10]:
    print(f'  {jf} -> {img}')
