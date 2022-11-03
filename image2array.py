# FILE NO # 3
# in this peice of code we convert our images into numpy array , and save them into .npy file
# we also cereat a lable array and also save it into .npy file
# we used the same code to create the test data set. we just changed the input to another folder.
import numpy as np
import os
from matplotlib.image import imread

#function to convert all images to array
path_cat = "kitty/"  # path of the cat pictures
path_dog = "DOG/"  # pathe to the dogs pics
animal = []                       # this is an empty list to append all pictures in it ( cats and dogs )
label = []                          # create a data set , and label [] is the label array

for file in os.listdir(path_cat): # here to iterate over all the pictures in the cat folder
                                    # we read them and convert them to array

    arr = imread(path_cat+file)
    arr = arr.flatten()             # read immage as array
    animal.append(arr)              # append the animal array with the image's array
    label.append(0)                 # we append the label to zero for cats


for file in os.listdir(path_dog):  # this is a repetition to the same loop above
    arr = imread(path_dog+file)    # here we do the same for dogs
    arr = arr.flatten()
    animal.append(arr)
    label.append(1)                     # we label the dogs with 1

# animal = NormalizeData(animal)
animal = np.array(animal)  # convert the list into array
label = np.array(label)     # convert the list into array

is_scalled= np.all((animal >= 0)) and np.all((animal <= 1))   # to check if all the values of the array between 0 and 1

print(is_scalled)


print(animal.shape)
# print(animal.size,animal.shape,animal.ndim) # print the sise , shape and dimention
# print(label.size,label.shape,label.ndim)
#
np.save("data_set",animal)  # here to save the data/ label into .npy array
np.save("labels",label)

