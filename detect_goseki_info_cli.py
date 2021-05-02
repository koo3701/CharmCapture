import argparse
import goseki_tools

parser = argparse.ArgumentParser()

parser.add_argument('path')

parser.add_argument('-t', '--type', default='renkin')

args = parser.parse_args()

path = args.path
type = args.type if args.type in ['renkin', 'box'] else 'renkin'

frames = goseki_tools.video2frame(path, 0.9, type)

for frame in frames:
  info = goseki_tools.get_goseki_info(frame, type)
  goseki_tools.print_goseki(info)
