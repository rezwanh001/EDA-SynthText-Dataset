SynthText in the Wild Dataset
-----------------------------
Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman
Visual Geometry Group, University of Oxford, 2016


Data format:
------------

SynthText.zip (size = 42074172 bytes (41GB)) contains 858,750 synthetic
scene-image files (.jpg) split into 200 directories, with 
7,266,866 word-instances, and 28,971,487 characters.

Ground-truth annotations are contained in the file "gt.mat" (Matlab format).
The file "gt.mat" contains the following cell-arrays, each of size 1x858750:

  1. imnames :  names of the image files

  2. wordBB  :  word-level bounding-boxes for each image, represented by
                tensors of size 2x4xNWORDS_i, where:
                   - the first dimension is 2 for x and y respectively,
                   - the second dimension corresponds to the 4 points
                     (clockwise, starting from top-left), and
                   -  the third dimension of size NWORDS_i, corresponds to
                      the number of words in the i_th image.

  3. charBB  : character-level bounding-boxes,
               each represented by a tensor of size 2x4xNCHARS_i
               (format is same as wordBB's above)

  4. txt     : text-strings contained in each image (char array).
               
               Words which belong to the same "instance", i.e.,
               those rendered in the same region with the same font, color,
               distortion etc., are grouped together; the instance
               boundaries are demarcated by the line-feed character (ASCII: 10)

               A "word" is any contiguous substring of non-whitespace
               characters.

               A "character" is defined as any non-whitespace character.


For any questions or comments, contact Ankush Gupta at:
removethisifyouarehuman-ankush@robots.ox.ac.uk

* Download Link: https://hyper.ai/datasets/4917



## Run File:

  `python3 char_count.py <SythText_datset_Path>`
  `#python3 char_count.py ./SynthText/SynthText/`

  `python3 synthtext.py <datapath_of_SynthText.zip> `
  `#python3 synthtext.py  ./SynthText.zip`

  `python3 convert2txt.py <datapath_of_SynthText_gt_mat.mat> `
  `#python3 convert2txt.py  ./SynthText/SynthText/SynthText/gt.mat`

