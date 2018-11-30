import numpy, os
import math, matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from scipy.signal import filtfilt

show_images = False

def show_image(array):
    if show_images:
        try:
            Image.fromarray(array).show()
        except:
            array.show()

#COMPLETE
def extract_head_ix_v2(y_RMS):
    m = max(y_RMS)

    head_max_ix = min([i for i, j in enumerate(y_RMS) if j == m])

    RMS_thresh = numpy.mean(y_RMS)*1.33

    RMS_diff = numpy.concatenate((0,numpy.diff(y_RMS)), axis=None)

    curr_ix = head_max_ix+1
    #print(y_RMS.shape)
    while (y_RMS[curr_ix] >= RMS_thresh):
        curr_ix = curr_ix+1
    #print(curr_ix)
    while (RMS_diff[curr_ix] < 0):
        curr_ix = curr_ix+1

    head_edge_ix = curr_ix-1

    return head_edge_ix-1


#Needs to be checked
#function [y_RMS_pos,y_RMS_neg] = calc_y_RMS_v2(im_input,x_center,y_center)
# v2 - split RMS calculation to positive and negatives halves
def calc_y_RMS_v2(im_input,x_center,y_center):
    s = im_input.shape
    row = s[0]
    col = s[1]
    y_RMS_neg = numpy.zeros(row)
    for ii in range(row):
        for jj in range(int(x_center)):
            if (im_input[ii,jj] > 0):
                y_RMS_neg[ii] = y_RMS_neg[ii] + (jj - x_center)**2
        y_RMS_neg[ii] = numpy.sqrt(y_RMS_neg[ii])

    y_RMS_pos = numpy.zeros(row)
    for ii in range(row):
        for jj in range(int(x_center+1),col):
            if (im_input[ii,jj] > 0):
                y_RMS_pos[0][ii] = y_RMS_pos[ii] + (jj - x_center)**2
        y_RMS_pos[ii] = numpy.sqrt(y_RMS_pos[ii])


    m = max(y_RMS_pos)
    max_ix = max([i for i, j in enumerate(y_RMS_pos) if j == m])
    if (max_ix > y_center):            # flip screw 180 degrees
        y_RMS_pos = numpy.flip(y_RMS_pos,axis = 0)
        y_RMS_neg = numpy.flip(y_RMS_neg,axis = 0)
    return y_RMS_pos,y_RMS_neg


def calc_y_outline(im_input,x_center,y_center):
    s = im_input.shape
    x_center = int(x_center)
    row = s[0]
    col = s[1]
    y_out_neg = numpy.zeros(row)
    y_out_pos = numpy.zeros(row)

    for ii in range(row):
        pixel_ix = numpy.where(im_input[ii, :] == 255)[0]
        if pixel_ix.size == 0:
            y_out_neg[ii] = 0
            y_out_pos[ii] = 0
        else:
            pixel_neg_ix = pixel_ix[pixel_ix <= x_center]
            if pixel_neg_ix.size == 0:
                y_out_neg[ii] = 0
            else:
                y_out_neg[ii] = x_center - min(pixel_neg_ix)
            pixel_pos_ix = pixel_ix[pixel_ix >= x_center]
            if pixel_pos_ix.size == 0:
                y_out_pos[ii] = 0
            else:
                y_out_pos[ii] = max(pixel_pos_ix) - x_center

    m = max(y_out_pos)
    max_ix = max([i for i, j in enumerate(y_out_pos) if j == m])
    if (max_ix > y_center):            # flip screw 180 degrees
        y_out_pos = numpy.flip(y_out_pos,axis = 0)
        y_out_neg = numpy.flip(y_out_neg,axis = 0)
    return y_out_pos,y_out_neg



#NOT COMPLETE
def estimate_parameters(im_fill,y_out,x_bounds,y_bounds,dpi):

    show_image(im_fill)

    smooth_length = 5
    y_out = y_out[y_out != 0]
    num_RMS_pts = len(y_out)

    y_RMS_smooth = filtfilt(numpy.ones(smooth_length),1,y_out)
    if show_images:
        plt.plot(y_RMS_smooth, 'k')
        plt.show()

    #Check below
    head_edge_ix = extract_head_ix_v2(y_RMS_smooth)

    pixel_width_v = numpy.zeros(int(y_bounds[1]-head_edge_ix))
    #fix y bounds
    n = 0
    for kx in range(int(y_bounds[0]+head_edge_ix),int(y_bounds[1])):
        pixel_width_v[n] = sum(im_fill[kx, :] > 0)
        n += 1
    #body_pixel_width = max(pixel_width_v)
    #screw_width = body_pixel_width/dpi

    body_pixel_length = y_bounds[1]-y_bounds[0] - head_edge_ix
    body_screw_length = body_pixel_length/dpi

    body_RMS_smooth = y_RMS_smooth[range(head_edge_ix,num_RMS_pts)] #y_RMS_smooth must be numpy.array

    if show_images:
        mn_rms = numpy.mean(body_RMS_smooth)
        plt.plot(body_RMS_smooth, 'k')
        plt.plot(numpy.ones(len(body_RMS_smooth))*mn_rms, "r--")
        plt.plot(numpy.ones(len(body_RMS_smooth))*mn_rms*1.05, "k--")
        plt.plot(numpy.ones(len(body_RMS_smooth))*mn_rms*0.95, "k--")
        plt.title("RMS_SMOOTH")
        plt.show()

    [thread_count,thread_spacing] = estimate_thread_count(body_RMS_smooth)
    print("Thread Count " + str(thread_count))
    thread_count = int(thread_count)
    body_outline = y_out[head_edge_ix:]
    thread_max_width_v = numpy.zeros((thread_count -1,1))
    pixel_spacing = round(thread_spacing)
    for r in range(thread_count-1):
        if r*pixel_spacing <= len(body_outline):
            temprange = range(int((r-1)*pixel_spacing+1),int(r*pixel_spacing))#check indexing
            thread_slice = body_outline[temprange]
            thread_max_width_v[r] = max(thread_slice)
        else:
            thread_max_width_v = numpy.delete(thread_max_width_v,r)

    body_pixel_width = numpy.median(thread_max_width_v)
    body_screw_width = 2*body_pixel_width/dpi

    head_outline = y_out[0:head_edge_ix]

    head_pixel_width = max(head_outline)
    max_head_ix = max([i for i, j in enumerate(head_outline) if j == head_pixel_width])

    head_screw_width = 2*head_pixel_width/dpi
    head_max_loc = max_head_ix/dpi
    total_screw_length = (y_bounds[1]-y_bounds[0])/dpi

    return total_screw_length,body_screw_length,body_screw_width,head_screw_width,head_max_loc,thread_count

#NOT COMPLETE
#function [thread_count,thread_spacing] = estimate_thread_count(body_RMS)
def estimate_thread_count(body_RMS):
# Input body RMS (smoothed or raw) and estimate thread_count based on crossing
# mean value of RMS
    mean_RMS_body = numpy.mean(body_RMS)#fix
    thread_count = 0
    thread_spacing = 0

    body_samples = len(body_RMS) - 1
    iterate = 1
    curr_ix = body_samples
    crit_point = mean_RMS_body*.95
    crit_point_ix = 0
    while iterate:
        if body_RMS[curr_ix] > crit_point:
            crit_point_ix = curr_ix
            iterate = 0
        else:
            curr_ix = curr_ix-1

    mean_RMS_body = numpy.mean(body_RMS[0:crit_point_ix]) #fix

    if (body_RMS[0] <= mean_RMS_body):
        below_thresh = 1
    else:
        below_thresh = 0

    pos_cross_v = []#fix
    neg_cross_v = []#fix
    for ii in range(len(body_RMS)):
        if below_thresh:
            if body_RMS[ii] > mean_RMS_body:
                below_thresh = 0
                pos_cross_v.append(ii)

        else:
            if body_RMS[ii] <= mean_RMS_body:
                below_thresh = 1
                neg_cross_v.append(ii)
    pos_spacing_v = numpy.zeros(len(pos_cross_v)-1)
    for kx in range(len(pos_cross_v)-1):
        pos_spacing_v[kx] = pos_cross_v[kx+1]-pos_cross_v[kx] #check

    median_pos_spacing = numpy.median(pos_spacing_v)
    if math.isnan(median_pos_spacing):
        median_pos_spacing = 0
    neg_spacing_v = numpy.zeros(len(neg_cross_v)-1)
    for kx in range(len(neg_cross_v)-1):
        neg_spacing_v[kx] = neg_cross_v[kx+1]-neg_cross_v[kx] #check
    median_neg_spacing = numpy.median(neg_spacing_v)
    thread_spacing = (median_pos_spacing+median_neg_spacing)/2
    thread_count = round(body_samples/thread_spacing)

    return thread_count,thread_spacing

#DONE
#function [im_out,x_center,y_center,x_bounds,y_bounds] = cleanup_image(im_input)
def cleanup_image(im_input):

    print("starting filtering")
    im_filter_1 = im_diff_filter(im_input, 14, [1, -1])
    im_filter_2 = im_diff_filter(im_input, 14, [1, 0, 0, -1])

    show_image(im_filter_1)
    show_image(im_filter_2)

    s = im_filter_1.shape
    both_pic = numpy.zeros((s[0], s[1]))
    both_pic[(im_filter_1 == 255) | (im_filter_2 == 255)] = 255

    im_filter = both_pic
    show_image(both_pic)

    im_crop, row_crop_ix, col_crop_ix, x_center, y_center, x_bounds, y_bounds = crop_image(im_filter, 0)
    show_image(im_crop)
    im_align, im_angle, slope_RMS = alignimage(im_crop) #probable bug in alignimage for 180 degrees

    show_image(im_align)

    # ============ Fill in image ==============
    im_fill,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds = crop_image(im_align,1) #check
    #im_crop,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds

    show_image(im_fill)

    # =============== Morphological Transform =================
    SE_grow = numpy.ones((3,3),numpy.uint8)
    SE_grow[0][0] = 0
    SE_grow[0][2] = 0
    SE_grow[2][2] = 0
    SE_grow[2][0] = 0
    SE_erode = numpy.ones((3,2),numpy.uint8)

    im_dilated = cv.dilate(im_fill,SE_grow,iterations = 1) #check
    show_image(im_dilated)
    im_eroded = cv.erode(im_dilated,SE_erode,iterations = 1) #check
    show_image(im_eroded)

    im_fill = im_eroded

    s = im_fill.shape
    yCoords, xCoords = numpy.nonzero(im_fill > 0)

    curRow = yCoords[0]
    start = 0
    stop = 0
    for y in range(len(yCoords)):
        if yCoords[y] == curRow and y != len(yCoords):
            continue
        else:
            stop = y-1
            im_fill[curRow, xCoords[start]:xCoords[stop]] = 255
            start = y
            curRow = yCoords[y]





    im_align_2,im_angle_2,slope_RMS = alignimage(im_fill) #check
    show_image(im_align_2)
    im_align_2,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds = crop_image(im_align_2,0)
    show_image(im_align_2)
    im_align_3 = Image.fromarray(im_align_2)
    dnc,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds = crop_image(im_align_3,1)

    im_out = im_align_2

    fileLocation = os.getcwd() + "\\" + "filled_image.bmp"
    print(fileLocation)
    result = Image.fromarray(im_out.astype(numpy.uint8))
    result.save(fileLocation)

    return im_out,x_center,y_center,x_bounds,y_bounds, fileLocation

#DONE
#function [im_crop,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds] = crop_image(im_align)
# =========== Calculate center of object =================
def crop_image(im_align,orient):
    #takes in vertical image of screw in binary form
    print("start crop")

    im_align = numpy.array(im_align)
    s = im_align.shape
    row = s[0]
    col = s[1]
    x_proj_v = numpy.arange(0, col)
    y_proj_v = numpy.arange(0, row)

    row_sums = numpy.sum(im_align, axis=0)
    col_sums = numpy.sum(im_align, axis=1)
    x_proj_v[row_sums == 0] = 0
    y_proj_v[col_sums == 0] = 0

    x_sep_block_ix = [] #fix
    on_block = 0
    num_blocks = 0

    for ii in range(len(x_proj_v)):
        if (on_block):
            if (x_proj_v[ii] == 0):
                on_block = 0
                x_sep_block_ix.append(x_proj_v[ii-1])

        else:
            if (x_proj_v[ii] != 0):
                on_block = 1
                x_sep_block_ix.append(x_proj_v[ii])
                num_blocks = num_blocks+1

    x_sep_block_ix = numpy.array(x_sep_block_ix)
    sep_block_lengths = x_sep_block_ix[range(1,len(x_sep_block_ix),2)]-x_sep_block_ix[range(0,len(x_sep_block_ix)-1,2)] #fix

    x_max_block_ix = numpy.argmax(sep_block_lengths)

    y_sep_block_ix = []#fix
    on_block = 0
    num_blocks = 0
    y_proj_v = y_proj_v
    for ii in range(len(y_proj_v)):
        if (on_block):
            if (y_proj_v[ii] == 0):
                on_block = 0
                y_sep_block_ix.append(y_proj_v[ii-1])

        else:
            if (y_proj_v[ii] != 0):
                on_block = 1
                y_sep_block_ix.append(y_proj_v[ii])
                num_blocks = num_blocks+1
    y_sep_block_ix = numpy.array(y_sep_block_ix)
    sep_block_lengths = y_sep_block_ix[range(1,len(y_sep_block_ix),2)]-y_sep_block_ix[range(0,len(y_sep_block_ix)-1,2)]#check
    y_max_block_ix = numpy.argmax(sep_block_lengths)

    x_pixel_length = x_sep_block_ix[(x_max_block_ix+1)*2 - 1] - x_sep_block_ix[(x_max_block_ix+1)*2 - 2]#CHECK WITH CHRIS
    y_pixel_length = y_sep_block_ix[(y_max_block_ix+1)*2 - 1] - y_sep_block_ix[(y_max_block_ix+1)*2 - 2]#CHECK WITH CHRIS

    x_proj_v[range(int(x_sep_block_ix[(x_max_block_ix+1)*2 - 2]))] = 0
    x_proj_v[int(x_sep_block_ix[(x_max_block_ix+1)*2-1]):] = 0
    y_proj_v[range(int(y_sep_block_ix[(y_max_block_ix+1)*2 - 2]))] = 0
    y_proj_v[int(y_sep_block_ix[(y_max_block_ix+1)*2 - 1]):] = 0

    x_proj_ix = x_proj_v[x_proj_v != 0]
    y_proj_ix = y_proj_v[y_proj_v != 0]

    x_center = numpy.mean(x_proj_ix)

    y_center = numpy.mean(y_proj_ix)

    #x center and y center are values within reason, will not match due to image rotation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    row_crop_ix = numpy.arange(int(y_center-round(y_pixel_length*.6)),int(y_center+round(y_pixel_length*.6)))#check
    #Check above before continuing
    ixr = row_crop_ix[(row_crop_ix >= 1) & (row_crop_ix <= row)]

    col_crop_ix = numpy.arange(int(x_center-x_pixel_length),int(x_center+x_pixel_length))  # check
    # Check above before continuing
    ixc = col_crop_ix[(col_crop_ix >= 1) & (col_crop_ix <= col)]

    ixc = range(int(ixc[0]),int(ixc[-1]))
    ixr = range(int(ixr[0]),int(ixr[-1]))

    x_proj_v = x_proj_v[ixc]
    y_proj_v = y_proj_v[ixr]

    x_proj_v = x_proj_v[x_proj_v > 0]
    y_proj_v = y_proj_v[y_proj_v > 0]

    x_bounds = numpy.concatenate((x_proj_v[0], x_proj_v[len(x_proj_v) - 1]), axis = None) #check
    y_bounds = numpy.concatenate((y_proj_v[0], y_proj_v[len(y_proj_v) - 1]), axis = None) #check

    im_align[range(int(y_bounds[0]))] = 0#check!!!!
    im_align[range(int(y_bounds[1]+1), len(im_align))] = 0#check MATLAB code carefully!!!

    im_align = im_align.T

    im_align[range(int(x_bounds[0]))] = 0#check!!!
    im_align[range(int(x_bounds[1]+1), len(im_align))] = 0#check MATLAB code carefully!!!
    im_align = im_align.T

    im_crop = im_align[ixr[0]:ixr[-1], ixc[0]:ixc[-1]]

    print("done crop")

    return im_crop,row_crop_ix,col_crop_ix,x_center,y_center,x_bounds,y_bounds

def get_parameters(file_loc):

    dpi = 326

    print("start")
    img = cv.imread(file_loc)
    print(file_loc)
    im_crop = img[27:1065,315:1760]
    im_crop = Image.fromarray(im_crop)

    im_fill,x_center,y_center,x_bounds,y_bounds, filledImageLocation = cleanup_image(im_crop)

    show_image(im_fill)

    y_out_pos,y_out_neg = calc_y_outline(im_fill,x_center,y_center)

    pos_accleration = sum(abs(numpy.diff(numpy.diff(y_out_pos))))
    neg_accleration = sum(abs(numpy.diff(numpy.diff(y_out_neg))))
    if pos_accleration <= neg_accleration:
        y_outline = y_out_pos
    else:
        y_outline = y_out_neg

    total_screw_length,body_screw_length,body_screw_width,head_screw_width,head_max_loc,thread_count = estimate_parameters(im_fill,y_outline,x_bounds,y_bounds,dpi)

    return total_screw_length,body_screw_length,body_screw_width,head_screw_width,head_max_loc,thread_count, filledImageLocation

def rotateImage(points,image,angle):
    angle = 90+angle
    angle = angle* -math.pi/180
    unitrotation = [[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]

    pointstranspose = points.T


    avgX = sum(pointstranspose[0])/len(pointstranspose[0])
    avgY = sum(pointstranspose[1])/len(pointstranspose[1])
    s = points.shape
    ptsShifta = numpy.ones((s[0],1))*avgX
    ptsShiftb = numpy.ones((s[0],1))*avgY

    ptsShift = numpy.concatenate((ptsShifta,ptsShiftb), axis=1)

    for i in range(s[0]):
        tempvec = numpy.matmul(unitrotation,[ [points[i][0]],[points[i][1]] ] )
        points[i][0] = tempvec[0]
        points[i][1] = tempvec[1]

    points = points + ptsShift


    py = image.shape
    hyp = round(((py[0]**2)+(py[1]**2))**0.5)
    rotatedImage = numpy.zeros((hyp,hyp),'float')
    adjrow = round((hyp - py[0])/2)
    adjcol =  round((hyp - py[1])/2)

    for x in range(py[0]):
        for y in range(py[1]):
            rotatedImage[x+adjrow][y+adjcol] = image[x][y]
    rotatedImage = Image.fromarray(rotatedImage)

    #Check how to rotate angle
    tt = rotatedImage.rotate(angle)

    image = numpy.array(tt,'float')
    s = image.shape
    image[image == 0] = 220


    return points,image



def rotate_image(im, angle):
    h, w = im.shape[:2]
    rot_mat = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv.warpAffine(im, rot_mat, (w, h), flags=2)

def getAngle(num):
    y = (180/(math.pi))*numpy.arctan(num)
    return y


def getAvgRMSE(pts, B):
    #pts = nx2 vector of x and y coords
    #B = [y intercept, slope]

    q = pts.shape
    #% get line standard form (ax + by + c = 0)
    a = B[0][1]
    b = -1
    c = B[0][0]

    d = numpy.zeros((q[0],1))
    l = len(d)
    #print(l)
    for x in range(l):
        rr = pnt2line(pts[x],a,b,c)

        d[x][0] = rr
    #print(d)
    d = d ** 2

    RMSE = (sum(d))**.5
    RMSE = RMSE /len(d)
    return RMSE

def pnt2line(pt, a, b, c):
    #pt = [x,y]
    #line = ax + by + c = 0

    x = pt[0]
    y = pt[1]

    d = abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
    return d

def getSlope(binaryImage):
    s = binaryImage.shape
    numPts = sum(sum(binaryImage == 255))
    yCoords = numpy.zeros((numPts, 1))
    xCoords = numpy.ones((numPts, 2))

    whereRes = numpy.where(binaryImage==255)
    yCoords[:] = whereRes[0].reshape(len(whereRes[0]), 1)
    xCoords[:, 1] = whereRes[1]
    B = numpy.linalg.lstsq(xCoords,yCoords,rcond = -1)

    slope = B[0][1]

    xCoords = xCoords.T
    xCoordsC = xCoords[1].T
    xCoordsC = xCoordsC.reshape(len(xCoordsC), 1)
    pts = numpy.concatenate((xCoordsC,yCoords),axis = 1)

    RMSE = getAvgRMSE(pts, B)

    return numpy.concatenate((RMSE[0],slope), axis = None)


def im_diff_filter(pic, thresh, h_filter):
    pic = pic.transpose(5)
    size = pic.size
    row = size[0]
    col = size[1]

    h_len = len(h_filter)
    cutout = int(h_len / 2)

    num_pic = numpy.array(pic, "double")
    num_pic = (num_pic[:,:,0] + num_pic[:,:,1] + num_pic[:,:,2])/3.0
    row_pic = numpy.zeros((col, row))
    col_pic = numpy.zeros((row, col))

    for r in range(col):
        # rr = (num_pic[c].T)
        row_diff = abs(numpy.convolve(num_pic[r], h_filter))
        row_diff = row_diff[int(cutout):int(len(row_diff) - cutout)]
        row_diff[0:h_len - 2] = 0
        row_diff[len(row_diff) - h_len + 1:len(row_diff)] = 0
        if h_len % 2 == 0:
            row_diff = numpy.concatenate((0, row_diff), axis=None)

        row_pic[r] = row_diff

    row_pic[row_pic > thresh] = 255
    row_pic[row_pic <= thresh] = 0

    num_pic = numpy.transpose(num_pic)
    for c in range(row):
        # col_diff = abs(numpy.diff(num_pic[r],1))
        col_diff = abs(numpy.convolve(num_pic[c], h_filter))
        col_diff = col_diff[int(cutout):int(len(col_diff) - cutout)]
        col_diff[0:h_len - 2] = 0
        col_diff[len(col_diff) - h_len + 1:len(col_diff)] = 0
        if h_len % 2 == 0:
            col_diff = numpy.concatenate((0, col_diff), axis=None)

        col_pic[c] = col_diff

    col_pic = numpy.transpose(col_pic)

    col_pic[col_pic > thresh] = 255
    col_pic[col_pic <= thresh] = 0

    s = col_pic.shape
    both_pic = numpy.zeros((s[0], s[1]))
    both_pic[(row_pic == 255) | (col_pic == 255)] = 255

    both_pic = both_pic.T

    return both_pic

def alignimage(im):
    print("start align")
    RMSESlope1 = getSlope(im)
    RMSESlope2 = getSlope(im.T)

    RMSE1 = RMSESlope1[0]
    RMSE2 = RMSESlope2[0]

    Slope1 = RMSESlope1[1]
    Slope2 = RMSESlope2[1]
    w = 0
    if RMSE1 <= RMSE2:
        slope = Slope1
        w = 1
    else:
        slope = 1 / Slope2

    if w == 0:
        RMSESlope = RMSESlope2
    else:
        RMSESlope = RMSESlope1

    angleScrew = getAngle(slope)


    x, y = im.shape


    hyp = round(((x**2)+(y**2))**0.5)

    top = bottom = (hyp-x)//2
    left = right = (hyp-y)//2
    rotatedImage = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])

    rotatedImage = Image.fromarray(rotatedImage)
    tt = rotatedImage.rotate(-90 + angleScrew)
    out = tt

    rotImArray = numpy.array(tt,"float")

    numPts = numpy.count_nonzero(rotImArray == 255)

    yCoords = numpy.zeros(numPts)
    xCoords = numpy.ones(numPts)

    s = rotImArray.shape
    count = 0
    yCoords, xCoords = numpy.nonzero(rotImArray==255)

    yCoordsSort = numpy.zeros(len(yCoords))

    indsY = yCoords.argsort(0)

    qq = 0
    for w in indsY:
        yCoordsSort[qq] = yCoords[w]
        qq = qq + 1

    xCoords2 = numpy.zeros(len(indsY))
    qq = 0

    for w in indsY:
        xCoords2[qq] = xCoords[w]
        qq = qq + 1
    uniqueY = numpy.unique(yCoords)

    RMSE = numpy.zeros(len(uniqueY))

    avgX = sum(xCoords)/len(xCoords)

    for i in range(len(uniqueY)):
        q = yCoords == uniqueY[i]
        q = q.T
        currentX = xCoords2[q]
        xDiff = currentX - avgX
        RMSE[i] = math.sqrt(sum(xDiff ** 2))

    m = max(RMSE)
    maxInd = max([i for i, j in enumerate(RMSE) if j == m])
    if (maxInd >= len(RMSE) / 2):
        out = tt.rotate(180)
    print("done align")
    return out,angleScrew,RMSESlope

