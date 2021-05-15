import os
import glob
import argparse
import charm_tools

parser = argparse.ArgumentParser()

parser.add_argument('directory')

parser.add_argument('-t', '--type', default='renkin')
parser.add_argument('-v', '--verbose', default=False, action='store_true')

args = parser.parse_args()

directory = args.directory
varbose = args.verbose
type = args.type if args.type in ['renkin', 'box', 'rinnne'] else 'renkin'

if varbose:
  num = 1
for path in glob.glob(os.path.join(directory, '*.mp4')):
  frames = charm_tools.video2frame(path, 0.9, type)

  if varbose:
    savepath = os.path.splitext(path)[0]
    os.makedirs(savepath, exist_ok=True)
  for frame in frames:
    info = charm_tools.get_charm_info(frame, type)
    charm_tools.print_charm(info)

    if varbose:
      filename = str(num) + '_' + \
        (info['skill'][0]['name'] if info['skill'][0]['name'] != '' else 'none') + '_' + \
        (info['skill'][0]['level'] if info['skill'][0]['level'] != '' else '0') + '_' + \
        (info['skill'][1]['name'] if info['skill'][1]['name'] != '' else 'none') + '_' + \
        (info['skill'][1]['level'] if info['skill'][1]['level'] != '' else '0') + '_' + \
        (info['slot'][0] if info['slot'][0] != '' else '0') + '_' + \
        (info['slot'][1] if info['slot'][1] != '' else '0') + '_' + \
        (info['slot'][2] if info['slot'][2] != '' else '0') + '.png'
      frame.save(os.path.join(savepath, filename))
      trims = charm_tools.trimming(os.path.join(savepath, filename), type)
      charm_tools.save(os.path.join(savepath, str(num)), trims)
      num += 1
