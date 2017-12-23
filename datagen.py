from Rhino.Display import *
import rhinoscriptsyntax as rs
import scriptcontext as sc
from System.Drawing import *
import random
import pickle
import math
import time

currentView = sc.doc.Views.ActiveView
picSize = Size(320,300)
baseGuid = "40e3aee1-36f4-4ec4-98a3-d21850bb50fd"
objectGuids = [
	"6152f52b-2556-42e1-9a16-1631bb38e311"
]

SCENE_OBJECTS = []
#obj = rs.GetObject()
#print(obj)
# returns the minimum and maximum of a bounding box.. a list of 2 points
def getMinMaxPts(bbox):
	inf = float("inf")
	minPt = [inf, inf , inf]
	maxPt = [-inf, -inf, -inf]
	
	for pt in bbox:
		for i in range(3):
			minPt[i] = min(minPt[i], pt[i])
			maxPt[i] = max(maxPt[i], pt[i])
	
	return [minPt, maxPt]

def reset_scene():
	global SCENE_OBJECTS
	rs.DeleteObjects(SCENE_OBJECTS)
	SCENE_OBJECTS = []

def placeObjectRandomly(objGuid):
	global baseGuid
	
	randAngle = random.sample([0,0.25,0.5,0.75],1)[0]
	randX = random.random()
	randY = random.random()
	
	scene_params = [randX, randY, randAngle]
	transformed_obj = transform_object(objGuid, scene_params)
	
	SCENE_OBJECTS.append(transformed_obj)

def transform_object(obj_id, scene_params):
	randX, randY, randAngle = scene_params
	#making a copy so that we dont mess with the original object
	obj_id = rs.CopyObject(obj_id)
	
	minObj, maxObj = getMinMaxPts(rs.BoundingBox([obj_id]))
	center = rs.VectorScale(rs.VectorAdd(minObj, maxObj),0.5)
	# print(randAngle)
	obj_id = rs.RotateObject(obj_id, center, randAngle*360)
	minObj, maxObj = getMinMaxPts(rs.BoundingBox([obj_id]))
	minBase, maxBase = getMinMaxPts(rs.BoundingBox([baseGuid]))
	
	objXW = maxObj[0] - minObj[0]
	objYW = maxObj[1] - minObj[1]
	
	xpos = randX*minBase[0] + (1-randX)*(maxBase[0] - objXW)
	ypos = randY*minBase[1] + (1-randY)*(maxBase[1] - objYW)
	
	translation = [xpos - minObj[0], ypos - minObj[1], minBase[2]-minObj[2]]
	return rs.MoveObject(obj_id, translation)

#return the view as a normalized float array
def getView():
		img = RhinoView.CaptureToBitmap(currentView, picSize)
		# img.Save("test.bmp")
		arr = [[n for n in range(picSize.Width)] for m in range(picSize.Height)]
		for y in range(picSize.Height):
			for x in range(picSize.Width):
				color = img.GetPixel(x,y)
				val = float((color.R + color.G + color.B)/3)/255
				arr[y][x] = val
				
		return arr

def create_scene():
	reset_scene()
	scene_params = placeObjectRandomly(objectGuids[0])
	return scene_params

def writeToFile(data, path):
	with open(path, 'wb') as output:
		pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	dsetSize = 20
	images = []
	answers = []
	for _ in range(dsetSize):
		rs.EnableRedraw(False)
		answers.append(create_scene())
		rs.EnableRedraw(True)
		images.append(getView())
		#time.sleep(1)
	
	writeToFile([images, answers], "dataset/0.pkl")
	reset_scene()