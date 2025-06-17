import sys
import os
import utils

if __name__ == "__main__":
    args = sys.argv[1:]
    threshold = 0.5
    video_path = ""
    if len(args) < 1:
        print("\nToo few arguments. You need to specify at least the path to the video. You can also specify the threshold between 0 and 1 (0.5 by default)")
        sys.exit(1)
    elif len(args) > 1:
        threshold = float(args[1])
        if threshold <= 0 or threshold >= 1:
            print("Threshold must be in interval (0, 1)")
            sys.exit(1)
    
    video_path = args[0]
    if not os.path.exists(video_path):
       print("File does not exist")
       sys.exit(1)

    utils.print_opening_interval(video_path, threshold)