import deeplabcut
import os
import time
import csv

prjexist = True
framesextracted, frameslabeled = True, True # tags for whether frames are already extracted or labeled (True) or not (False)
traindatasetexist = True # traindatasetexist for whether train dateset exist (True) or not (False)
tag_train = False # need to train or not
tag_evaluation = False # tags for whether evaluation and anlyze functions are needed (True) or not (False)
tag_analyze, tag_createlabel = True, True # tags for whether anlyze and createlabel functions are needed (True) or not (False)


task = 'gaitdatatestv2' # use the videos of dlc examples
experimenter = 'Jordan'

if prjexist:
	# project already exist
	print("Use exist project!!!")
	path_prj = os.path.join(os.getcwd(),'OurPrj',task + '-'+ experimenter + '-2019-04-16')
	path_config_file = os.path.join(path_prj, 'config.yaml')
else:
	video = os.path.join(os.getcwd(),'OurPrj',task + '-'+ experimenter + '-2019-04-16', 'videos','Az_Neural.avi')
	path_config_file = deeplabcut.create_new_project(task, experimenter, video, working_directory = working_dir, copy_videos = True)
	path_prj, config_file_name = os.path.split(path_config_file)

if not framesextracted:
	print("Extracting frame from videos")
	deeplabcut.extract_frames(path_config_file,'automatic','uniform',crop=False)
else:
	print("frames have been extracted already!!")

if not frameslabeled:
	print("Need labeled frames")
	deeplabcut.label_frames(path_config_file) # label frames
else:
	print("Frames have been labeled already!!!!!")


if not traindatasetexist: # create training dataset if not exist
	deeplabcut.check_labels(path_config_file)
	deeplabcut.create_training_dataset(path_config_file, num_shuffles=1)
else:
	print("Training dataset exists")

if tag_train:
	# train data argments, saveiters, displayiters, maxiters are the same as in pose_cfg.yaml, here are for visualization
	saveiters = 10000
	displayiters = 1000
	maxiters = 100000
	train_start = time.time()
	deeplabcut.train_network(path_config_file, shuffle=1, saveiters = saveiters, displayiters = displayiters, maxiters = maxiters)
	train_end = time.time()
	train_time = train_end - train_start
	print('maxiters is %d' %maxiters)
	print('training time is %f' %train_time)
	with open(os.path.join(path_prj,'traintime.csv'), 'w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',')
		csv_writer.writerow(['maxiters','training time'])
		csv_writer.writerow([maxiters,train_time])
else:
	print("No need to train!")

if tag_evaluation:
	deeplabcut.evaluate_network(path_config_file, plotting = True) # evaluation
else:
	print("No need to evaluate!")


if tag_analyze or tag_createlabel: 
	animal = 'bug'
	newvideo_path = os.path.join(path_prj, 'videostoanalyze', animal) 
	newvideos = [os.path.join(newvideo_path, 'CPB15_120418.mp4')]

	# animal = 'Az'
 #    # Az/
	# newvideo_path = os.path.join(path_prj, 'videostoanalyze', animal) 
	# newvideos = [os.path.join(newvideo_path, 'Az_Neural-190128_Azula-190128-141757_Cam1.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190313_Azula-190313-124618_Cam1.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190410_Azula-190410-120937_Cam1.avi')] 

	# # Az/turningbehavior
	# newvideo_path = os.path.join(path_prj, 'videostoanalyze', animal,'turningbehavior') 
	# newvideos = [os.path.join(newvideo_path, 'Az_Neural-190128_Azula-190128-141757_Cam2.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190228_Azula-190228-121940_Cam2.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190228_Azula-190228-123038_Cam2.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190410_Azula-190410-120937_Cam2.avi'), \
	# 			 os.path.join(newvideo_path, 'Az_Neural-190410_Azula-190410-122407_Cam2.avi')] 
	if tag_analyze:
		# analyze new video
		deeplabcut.analyze_videos(path_config_file, newvideos,save_as_csv = True, videotype = '.mp4')

	if tag_createlabel:
		# create labeled video in .mp4
		deeplabcut.create_labeled_video(path_config_file,newvideos)
	
# plot trajectories
#deeplabcut.plot_trajectories(path_config_file, newvideos)
