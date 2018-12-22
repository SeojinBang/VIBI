#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 22:34:27 2018

@author: seojinb
"""

#import PIL
#from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import re
#%%
review = batch.text
label = batch.label[0]
#%%
save_batch(dataset = self.dataset, batch = x, index = index_chunk, filename = img_name, is_cuda = self.cuda)##
#%%
text_rev = data.ReversibleField(tokenize = tokenizer_twolevel, batch_first = True)
#%%  
word_idx = self.word_idx          
width = 100
height = 100
current_line = 0 
skip_line = 10

img = Image.new("RGBA", (700, 1100), 'white')
draw = ImageDraw.Draw(img)

for i in range(batch.size(0)):
    
    current_line = current_line + skip_line * 2
    label_review = "POS" if label[i] is 3 else "NEG"
    review = idxtoreview(batch[i])
    review_selected = idxtoreview(batch[i], index)
    draw.text((20, current_line), "sentiment: " + label_review, 'blue')
    
    num_line = len(review) // width
    
    for j in range(num_line):
        
        current_line = current_line + skip_line
        draw.text((20, current_line), review[(width * j):(width * (j + 1))], 'red')    
        draw.text((20, current_line), review_selected[(width * j):(width * (j + 1))], 'black')
        
draw = ImageDraw.Draw(img)            
img.save("test.png")


def idxtoreview(review, index = None, word_idx):
    
    review = np.array(word_idx)[review.tolist()]
    review = [re.sub(r"<pad>", "", review_sub) for review_sub in review]
    review = [re.sub(' +', ' ', review_sub) for review_sub in review]
    review = [review_sub.strip() for review_sub in review]
    
    if index is not None:    

        review_selected = [len(review_sub) * "_" for review_sub in review]
        review_selected[index] = review[index]
        review = review_selected
    
    review = " ".join(review)
    review = re.sub(' +', ' ', review)
    review = review.strip()
    
    return review


