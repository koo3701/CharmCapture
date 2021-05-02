import os
import glob
import numpy as np
import cv2
from PIL import Image

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

def compare_image(l, r):
  threshold = 150
  m = 255
  w = 0

  l = l.convert('L')
  l = np.array(l)
  l[0][0] = m
  l = l > threshold
  t = np.where(np.any(l, axis=0))[0]
  t = t[-1] + 1 if len(t) > 0 else l.shape[1]
  w = max(w, t)

  r = r.convert('L')
  r = np.array(r)
  r[0][0] = m
  r = r > threshold
  t = np.where(np.any(r, axis=0))[0]
  t = t[-1] + 1 if len(t) > 0 else r.shape[1]
  w = max(w, t)

  l = l[:, :w]
  r = r[:, :w]
  o = l | r
  a = l & r

  l = l * m
  r = r * m
  o = o * m
  a = a * m

  return np.sum((l * o) / (np.linalg.norm(l) * np.linalg.norm(o))) * \
         np.sum((r * o) / (np.linalg.norm(r) * np.linalg.norm(o))) * \
         np.sum((l * a) / (np.linalg.norm(l) * np.linalg.norm(a))) * \
         np.sum((r * a) / (np.linalg.norm(r) * np.linalg.norm(a))) * \
         np.sum((o * a) / (np.linalg.norm(o) * np.linalg.norm(a)))



def goseki(skill1, level1, skill2, level2, slot1, slot2, slot3):
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
    base = (1035, 200)
  else:
    base = (772, 212)
  if isinstance(path, str):
    sample = Image.open(path)
  else:
    sample = path

  posisions = [
    (
      p[0] / 1280 * sample.width,
      p[1] / 720 * sample.height,
      (p[0] + p[2]) / 1280 * sample.width,
      (p[1] + p[3]) / 720 * sample.height,
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

  return goseki(*[sample.crop(p) for p in posisions])

resource = 'slot/'
slot = {}
for path in glob.glob(os.path.join(resource, '*.png')):
  slot[os.path.splitext(os.path.basename(path))[0]] = Image.open(path).convert('L')

def get_slot(trims):
  res = []
  for s in trims['slot']:
    r = {}
    for level, pict in slot.items():
      r[level] = compare_image(s, pict)
    res.append(max(r, key=r.get))
  
  return res

resource = 'skill/'
skill_name = {}
for path in glob.glob(os.path.join(resource, '*.png')):
  skill_name[os.path.splitext(os.path.basename(path))[0]] = Image.open(path).convert('L')

def get_skill_name(trims):
  res = []
  for s in trims['skill']:
    s = s['name']
    r = {}
    for level, pict in skill_name.items():
      r[level] = compare_image(s, pict)
    res.append(max(r, key=r.get))
  
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
  skill_level[os.path.splitext(os.path.basename(path))[0]] = Image.open(path).convert('L')

def get_skill_level(trims):
  res = []
  for s in trims['skill']:
    s = s['level']
    r = {}
    for level, pict in skill_level.items():
      r[level] = compare_image(s, pict)
    res.append(max(r, key=r.get))
  
  return res

def video2frame(path, threshold = 0.9, box = 'renkin'):
  if box == 'box':
    box = [633, 359, 364, 241]
  else:
    box = (287, 118, 448, 426)
  cap = cv2.VideoCapture(path)

  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  box = (box[0] / 1280 * width, box[1] / 720 * height, (box[0] + box[2]) / 1280 * width, (box[1] + box[3]) / 720 * height)

  first = True
  prev = None
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      frame = cv2pil(frame)
      if first:
        first = False
        prev = frame
        yield frame
        continue

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
      prev = frame
    else:
      break

def get_goseki_info(frame, type='renkin'):
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

  return goseki(skill[0], level[0], skill[1], level[1], slot[0], slot[1], slot[2])


def print_goseki(goseki):
  print('\t'.join([
    goseki['skill'][0]['name'],
    goseki['skill'][0]['level'],
    goseki['skill'][1]['name'],
    goseki['skill'][1]['level'],
    goseki['slot'][0],
    goseki['slot'][1],
    goseki['slot'][2],
  ]))

def output_goseki(filename, goseki):
  with open(filename, 'a', encoding='utf-8') as f:
    f.write('\t'.join([
      goseki['skill'][0]['name'],
      goseki['skill'][0]['level'],
      goseki['skill'][1]['name'],
      goseki['skill'][1]['level'],
      goseki['slot'][0],
      goseki['slot'][1],
      goseki['slot'][2],
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