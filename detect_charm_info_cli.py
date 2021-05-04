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
type = args.type if args.type in ['renkin', 'box'] else 'renkin'

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
      frame.save(os.path.join(savepath, str(num) + '.png'))
      trims = charm_tools.trimming(os.path.join(savepath, str(num) + '.png'))
      charm_tools.save(os.path.join(savepath, str(num)), trims)
      num += 1
