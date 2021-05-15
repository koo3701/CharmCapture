import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageFilter

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def compare_image(l, r, performance=False):
  threshold = 150
  m = 255
  w = 0

  # l = l.convert('L')
  l = np.array(l)
  l = l[:, :, 0]
  l[0][0] = m
  l = l > threshold
  t = np.where(np.any(l, axis=0))[0]
  t = t[-1] + 1 if len(t) > 0 else l.shape[1]
  w = max(w, t)

  # r = r.convert('L')
  r = np.array(r)
  r = r[:, :, 0]
  r[0][0] = m
  r = r > threshold
  t = np.where(np.any(r, axis=0))[0]
  t = t[-1] + 1 if len(t) > 0 else r.shape[1]
  w = max(w, t)

  l = l[:, :w]
  r = r[:, :w]
  if performance:
    o = l | r
    a = l & r
  x = l ^ r
  z = np.zeros(l.shape)

  if performance:
    o = o * m
    a = a * m
  x = x * m

  if performance:
    o = np.array(Image.fromarray(o.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)))
    a = np.array(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)))
  x = np.array(Image.fromarray(x.astype(np.uint8)).filter(ImageFilter.MedianFilter(3)))

  if performance:
    o[0][0] = m
    a[0][0] = m
  x[0][0] = m
  z[0][0] = m

  # Image.fromarray(l.astype(np.uint8)).save('l.png')
  # Image.fromarray(r.astype(np.uint8)).save('r.png')
  # Image.fromarray(o.astype(np.uint8)).save('o.png')
  # Image.fromarray(a.astype(np.uint8)).save('a.png')

  if performance:
    oa = np.sum((o * a) / (np.linalg.norm(o) * np.linalg.norm(a)))
  else:
    oa = 1
  return np.sum((x * z) / (np.linalg.norm(x) * np.linalg.norm(z))) * oa



def charm(skill1, level1, skill2, level2, slot1, slot2, slot3):
  return {
    'skill': (
      {
        'name': skill1,
        'level': level1,
      },
      {
        'name': skill2,
        'level': level2,
      }
    ),
    'slot': (
      slot1,
      slot2,
      slot3,
    )
  }

def trimming(path, base = 'renkin'):
  if base == 'box':
    base = (1034, 200)
  elif base == 'rinnne':
    base = (699, 212)
  else:
    base = (772, 212)
  if isinstance(path, str):
    sample = Image.open(path)
  else:
    sample = path
  

  if sample.width != 1280 or sample.height != 720:
    sample = sample.resize((1280, 720), Image.LANCZOS)

  posisions = [
    (
      p[0],
      p[1],
      p[0] + p[2],
      p[1] + p[3],
    ) for p in [
      [  0 + base[0],  65 + base[1], 214, 23], # skill 1
      [201 + base[0],  92 + base[1],  15, 20], # level 1
      [  0 + base[0], 116 + base[1], 214, 23], # skill 2
      [201 + base[0], 143 + base[1],  15, 20], # level 2
      [131 + base[0],   0 + base[1],  28, 25], # slot 1
      [159 + base[0],   0 + base[1],  28, 25], # slot 2
      [187 + base[0],   0 + base[1],  28, 25], # slot 3
    ]
  ]

  return charm(*[sample.crop(p) for p in posisions])

resource = 'slot/'
slot = {}
for path in glob.glob(os.path.join(resource, '*.png')):
  slot[os.path.splitext(os.path.basename(path))[0]] = Image.open(path)

def get_slot(trims):
  res = []
  for s in trims['slot']:
    m = 0
    res.append('none')
    for text, pict in slot.items():
      diff = compare_image(s, pict)
      if m < diff:
        m = diff
        res[-1] = text
  
  return res

resource = 'skill/'
skill_name = {}
for path in glob.glob(os.path.join(resource, '*.png')):
  skill_name[os.path.splitext(os.path.basename(path))[0]] = Image.open(path)

def get_skill_name(trims):
  res = []
  near = [
    (['氷属性攻撃強化', '水属性攻撃強化'], (0, 0.4)),
    (['火属性攻撃強化', '水属性攻撃強化'], (0, 1)),
    (['氷耐性', '水耐性'], (0, 0.4)),
    (['火耐性', '水耐性'], (0, 1)),
    (['氷耐性', '毒耐性'], (0.7, 1)),
    (['雷属性攻撃強化', '龍属性攻撃強化'], (0, 1)),
    (['雷耐性', '龍耐性'], (0, 1)),
    (['属性やられ耐性', '爆破やられ耐性'], (0, 2)),
    (['体術', '陽動'], (0, 1)),
    (['体術', '業物'], (0.5, 1.0)),
    (['防御', '耐震'], (0, 1)),
    (['不屈', '逆襲'], (0, 1)),
    (['逆恨み', '逆襲'], (2, 3)),
    (['睡眠耐性', '麻痺耐性'], (0, 2)),
    (['睡眠属性強化', '爆破属性強化'], (0, 1)),
    (['麻痺属性強化', '爆破属性強化'], (0.5, 1.5)),
    (['翔蟲使い', '鈍器使い'], (0, 2)),
    (['ブレ抑制', '反動軽減'], (0, 2)),
    (['地質学', '植生学'], (0, 2)),
    (['ランナー', '達人芸'], (1, 2)),
    # (['貫通弾・貫通矢強化', '通常弾・連射矢強化'], (0, 2)),
    (['抜刀術【技】', '抜刀術【力】'], (4, 5)),
  ]
  width = 17
  for s in trims['skill']:
    s = s['name']
    m = 0
    res.append('none')
    for text, pict in skill_name.items():
      diff = compare_image(s, pict)
      if m < diff:
        m = diff
        res[-1] = text
    for text, size in near:
      if res[-1] in text:
        area = (width * size[0], 0, width * size[1], s.height)
        m = 0
        l = s.crop(area)
        for t in text:
          r = skill_name[t].crop(area)
          diff = compare_image(l, r, True)
          if m < diff:
            m = diff
            res[-1] = t


  
  return res

# resource = 'skill/'
# skill_name = {}
# for path in glob.glob(os.path.join(resource, '*.png')):
#   skill_name[os.path.splitext(os.path.basename(path))[0]] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# def get_skill_name(trims):
#   res = []
#   for s in trims['skill']:
#     s = s['name']
#     s = pil2cv(s)
#     s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
#     r = {}
#     for level, pict in skill_name.items():
#       r[level] = cv2.matchTemplate(s, pict, cv2.TM_CCOEFF_NORMED)
#     res.append(max(r, key=r.get))
  
#   return res

resource = 'level/'
skill_level = {}
for path in glob.glob(os.path.join(resource, '*.png')):
  skill_level[os.path.splitext(os.path.basename(path))[0]] = Image.open(path)

def get_skill_level(trims):
  res = []
  for s in trims['skill']:
    s = s['level']
    m = 0
    res.append('none')
    for text, pict in skill_level.items():
      diff = compare_image(s, pict)
      if m < diff:
        m = diff
        res[-1] = text
  
  return res

def video2frame(path, threshold = 0.9, box = 'renkin'):
  if box == 'box':
    boxes = [[633, 359, 364, 241], [785, 576, 23, 26]]
  elif box == 'rinnne':
    boxes = [[320, 366, 331, 169], [458, 535, 25, 28]]
  else:
    boxes = [[287, 118, 448, 426]]
  cap = cv2.VideoCapture(path)

  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in boxes]

  prev = None
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      frame = cv2pil(frame)
      if frame.width != 1280 or frame.height != 720:
        frame = frame.resize((1280, 720), Image.LANCZOS)
      if prev is None:
        prev = frame
        yield frame
        continue

      for box in boxes:
        p = prev.crop(box)
        p = p.convert('L')
        p = np.array(p)
        p[0][0] = 128
        p = (p > 127) * 255

        n = frame.crop(box)
        n = n.convert('L')
        n = np.array(n)
        n[0][0] = 128
        n = (n > 127) * 255

        diff = np.sum((p / (np.linalg.norm(p))) * (n / (np.linalg.norm(n))))
        if diff < threshold:
          yield frame
          break
      prev = frame
    else:
      break

def get_charm_info(frame, type='renkin'):
  trims = trimming(frame, type)

  skill = get_skill_name(trims)

  skill[0] = skill[0] if skill[0] != 'none' else ''
  skill[1] = skill[1] if skill[1] != 'none' else ''

  level = get_skill_level(trims)

  level[0] = level[0] if level[0] != 'none' else ''
  level[1] = level[1] if level[1] != 'none' else ''

  slot = get_slot(trims)

  slot[0] = slot[0] if slot[0] != 'none' else ''
  slot[1] = slot[1] if slot[1] != 'none' else ''
  slot[2] = slot[2] if slot[2] != 'none' else ''

  return charm(skill[0], level[0], skill[1], level[1], slot[0], slot[1], slot[2])


def print_charm(charm):
  print('\t'.join([
    charm['skill'][0]['name'],
    charm['skill'][0]['level'],
    charm['skill'][1]['name'],
    charm['skill'][1]['level'],
    charm['slot'][0],
    charm['slot'][1],
    charm['slot'][2],
  ]))

def output_charm(filename, charm):
  with open(filename, 'a', encoding='utf-8') as f:
    f.write('\t'.join([
      charm['skill'][0]['name'],
      charm['skill'][0]['level'],
      charm['skill'][1]['name'],
      charm['skill'][1]['level'],
      charm['slot'][0],
      charm['slot'][1],
      charm['slot'][2],
    ]))
    f.write('\n')


def save(path, trims):
  os.makedirs(path, exist_ok=True)

  res = get_skill_name(trims)

  trims['skill'][0]['name'].save(os.path.join(path, 'skill1_name_' + str(res[0]) + '.png'))
  trims['skill'][1]['name'].save(os.path.join(path, 'skill2_name_' + str(res[1]) + '.png'))

  res = get_skill_level(trims)

  trims['skill'][0]['level'].save(os.path.join(path, 'skill1_level_' + str(res[0]) + '.png'))
  trims['skill'][1]['level'].save(os.path.join(path, 'skill2_level_' + str(res[1]) + '.png'))

  res = get_slot(trims)

  trims['slot'][0].save(os.path.join(path, 'slot1_' + str(res[0]) + '.png'))
  trims['slot'][1].save(os.path.join(path, 'slot2_' + str(res[1]) + '.png'))
  trims['slot'][2].save(os.path.join(path, 'slot3_' + str(res[2]) + '.png'))