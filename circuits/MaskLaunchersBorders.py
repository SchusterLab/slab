from slab.circuits.mp_components import *


def struct_startings(chip):

    struct_starting = list()
    struct_starting.append((700, chip.size[1] / 2 - 1200))
    struct_starting.append((700, chip.size[1] / 2 + 1000))
    struct_starting.append((chip.size[0] / 2 - 1900, chip.size[1] - 700))
    struct_starting.append((chip.size[0] / 2, chip.size[1] - 700))
    struct_starting.append((chip.size[0] / 2 + 1900, chip.size[1] - 700))
    struct_starting.append((chip.size[0] - 700, chip.size[1] / 2 + 1000))
    struct_starting.append((chip.size[0] - 700, chip.size[1] / 2 - 1000))
    struct_starting.append((chip.size[0] / 2 + 1900, 700))
    struct_starting.append((chip.size[0] / 2, 700))
    struct_starting.append((chip.size[0] / 2 - 1900, 700))
    return struct_starting


def chipInit(chip, defaults):
    """
    This makes the launch pads on the chip. Input is an object c, which is the chip.
    From the launch pads, we can make connections on the chip.
    """
    #The following creates 8 launch pads on the chip. There are 3 pads per side, two
    #in each of the corners and then one in the middle of each side.

    struct_starting = struct_startings(chip)

    setattr(chip, 'struct0', Structure(chip, start=struct_starting[0], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct1', Structure(chip, start=struct_starting[1], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct2', Structure(chip, start=struct_starting[2], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct3', Structure(chip, start=struct_starting[3], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct4', Structure(chip, start=struct_starting[4], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct5', Structure(chip, start=struct_starting[5], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct6', Structure(chip, start=struct_starting[6], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct7', Structure(chip, start=struct_starting[7], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct8', Structure(chip, start=struct_starting[8], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct9', Structure(chip, start=struct_starting[9], direction=90, defaults=defaults, layer = '0'))

def chipInit_10by20mm(chip, defaults):
    """
    This makes the launch pads on the chip. Input is an object c, which is the chip.
    From the launch pads, we can make connections on the chip.
    """
    #The following creates 8 launch pads on the chip. There are 3 pads per side, two
    #in each of the corners and then one in the middle of each side.

    struct_starting = struct_startings_10by20mm(chip)

    setattr(chip, 'struct0', Structure(chip, start=struct_starting[0], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct1', Structure(chip, start=struct_starting[1], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct2', Structure(chip, start=struct_starting[2], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct3', Structure(chip, start=struct_starting[3], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct4', Structure(chip, start=struct_starting[4], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct5', Structure(chip, start=struct_starting[5], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct6', Structure(chip, start=struct_starting[6], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct7', Structure(chip, start=struct_starting[7], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct8', Structure(chip, start=struct_starting[8], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct9', Structure(chip, start=struct_starting[9], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct10', Structure(chip, start=struct_starting[10], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct11', Structure(chip, start=struct_starting[11], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct12', Structure(chip, start=struct_starting[12], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct13', Structure(chip, start=struct_starting[13], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct14', Structure(chip, start=struct_starting[14], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct15', Structure(chip, start=struct_starting[15], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct16', Structure(chip, start=struct_starting[16], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct17', Structure(chip, start=struct_starting[17], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct18', Structure(chip, start=struct_starting[18], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct19', Structure(chip, start=struct_starting[19], direction=0, defaults=defaults, layer = '0'))

def chipInit_10by20mmwide(chip, defaults):
    """
    This makes the launch pads on the chip. Input is an object c, which is the chip.
    From the launch pads, we can make connections on the chip.
    """
    #The following creates 8 launch pads on the chip. There are 3 pads per side, two
    #in each of the corners and then one in the middle of each side.

    struct_starting = struct_startings_10by20mmwide(chip)
    setattr(chip, 'struct0', Structure(chip, start=struct_starting[0], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct1', Structure(chip, start=struct_starting[1], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct2', Structure(chip, start=struct_starting[2], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct3', Structure(chip, start=struct_starting[3], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct4', Structure(chip, start=struct_starting[4], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct5', Structure(chip, start=struct_starting[5], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct6', Structure(chip, start=struct_starting[6], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct7', Structure(chip, start=struct_starting[7], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct8', Structure(chip, start=struct_starting[8], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct9', Structure(chip, start=struct_starting[9], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct10', Structure(chip, start=struct_starting[10], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct11', Structure(chip, start=struct_starting[11], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct12', Structure(chip, start=struct_starting[12], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct13', Structure(chip, start=struct_starting[13], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct14', Structure(chip, start=struct_starting[14], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct15', Structure(chip, start=struct_starting[15], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct16', Structure(chip, start=struct_starting[16], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct17', Structure(chip, start=struct_starting[17], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct18', Structure(chip, start=struct_starting[18], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct19', Structure(chip, start=struct_starting[19], direction=0, defaults=defaults, layer = '0'))


#Rotated 10mm chip
def struct_startings_10mm(chip):

    struct_starting = list()

    #struct0
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 1, chip.size[1]-800))
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 2, chip.size[1]-800))
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 3, chip.size[1]-800))
    #
    # struct_starting.append((chip.size[0] - 800, chip.size[1]/2))
    #
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 3, 800))
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 2, 800))
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 1, 800))
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 0, 800))
    #
    # struct_starting.append((800, chip.size[1]/2))
    #
    # struct_starting.append((1200 + (chip.size[0] - 1200 * 2) / 3 * 0, chip.size[1] - 800))



    struct_starting.append((600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 1))
    struct_starting.append((600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 2))
    struct_starting.append((600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 3))

    struct_starting.append((chip.size[0] / 2, chip.size[1] - 600))

    struct_starting.append((chip.size[0] - 600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 3))
    struct_starting.append((chip.size[0] - 600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 2))
    struct_starting.append((chip.size[0] - 600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 1))
    struct_starting.append((chip.size[0] - 600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 0))

    struct_starting.append((chip.size[0] / 2, 600))

    struct_starting.append((600, 1200 + (chip.size[1] - 1200 * 2) / 3 * 0))

    return struct_starting
def struct_startings_10by20mm(chip):


    # create struct list for launcher locations for 10x20mm chip
    struct_starting = list()
    #struct_starting.append((chip.size[0] / 2, 600))
    top = chip.size[1]-600
    bottom = 600
    left = 600
    right = chip.size[0] - 600
    xcenter = chip.size[0]/2
    ycenter = chip.size[1]/2
    height = top-bottom
    length = right-left
    struct_starting.append((xcenter - length/3, top))
    struct_starting.append((xcenter, top))
    struct_starting.append((xcenter + length/3, top))
    for ii in range(8):
        if (ii>=1 and ii<=7):
            struct_starting.append((right,top - height*ii/8))
    struct_starting.append((xcenter - length / 3, bottom))
    struct_starting.append((xcenter, bottom))
    struct_starting.append((xcenter + length / 3, bottom))
    for ii in range(8):
        if (ii>=1 and ii<=7):
            struct_starting.append((left,bottom+height*ii/8))
    #print("HOW MANY STRUCTS: ")
    #print(" THIS MANY STRUCTS: %s" % str(len(struct_starting)))
    return struct_starting
def struct_startings_10by20mmwide(chip):


    # create struct list for launcher locations for 10x20mm chip
    struct_starting = list()
    #struct_starting.append((chip.size[0] / 2, 600))
    top = chip.size[1]-600
    bottom = 600
    left = 600
    right = chip.size[0] - 600
    xcenter = chip.size[0]/2
    ycenter = chip.size[1]/2
    height = top-bottom
    length = right-left
    for ii in range(8):
        if (ii>=1 and ii<=7):
            if ii==1 or ii==2:
                temp = (left + length * ii / 8 -450-600, top)
            elif ii==6 or ii==7:
                temp = (left + length * ii / 8 +450+600, top)
            else:
                temp = (left + length * ii / 8, top)
            struct_starting.append(temp)


    struct_starting.append((right, ycenter + height/3))
    struct_starting.append((right, ycenter))
    struct_starting.append((right, ycenter - height/3))
    for ii in range(8):
        if (ii >= 1 and ii <= 7):
            struct_starting.append((right - length * ii / 8, bottom))
    struct_starting.append((left, ycenter - height / 3))
    struct_starting.append((left, ycenter))
    struct_starting.append((left, ycenter + height / 3))

    #print("HOW MANY STRUCTS: ")
    #print(" THIS MANY STRUCTS: %s" % str(len(struct_starting)))
    return struct_starting


def chipInit_10mm(chip, defaults):
    """
    This makes the launch pads on the chip. Input is an object c, which is the chip.
    From the launch pads, we can make connections on the chip.
    """
    #The following creates 8 launch pads on the chip. There are 3 pads per side, two
    #in each of the corners and then one in the middle of each side.

    struct_starting = struct_startings_10mm(chip)

    setattr(chip, 'struct0', Structure(chip, start=struct_starting[0], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct1', Structure(chip, start=struct_starting[1], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct2', Structure(chip, start=struct_starting[2], direction=0, defaults=defaults, layer = '0'))
    setattr(chip, 'struct3', Structure(chip, start=struct_starting[3], direction=270, defaults=defaults, layer = '0'))
    setattr(chip, 'struct4', Structure(chip, start=struct_starting[4], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct5', Structure(chip, start=struct_starting[5], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct6', Structure(chip, start=struct_starting[6], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct7', Structure(chip, start=struct_starting[7], direction=180, defaults=defaults, layer = '0'))
    setattr(chip, 'struct8', Structure(chip, start=struct_starting[8], direction=90, defaults=defaults, layer = '0'))
    setattr(chip, 'struct9', Structure(chip, start=struct_starting[9], direction=0, defaults=defaults, layer = '0'))

#Rotated 10mm chip
# def chipInit_10mm(chip, defaults):
#     """
#     This makes the launch pads on the chip. Input is an object c, which is the chip.
#     From the launch pads, we can make connections on the chip.
#     """
#     #The following creates 8 launch pads on the chip. There are 3 pads per side, two
#     #in each of the corners and then one in the middle of each side.
#
#     struct_starting = struct_startings_10mm(chip)
#
#     setattr(chip, 'struct0', Structure(chip, start=struct_starting[0], direction=270, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct1', Structure(chip, start=struct_starting[1], direction=270, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct2', Structure(chip, start=struct_starting[2], direction=270, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct3', Structure(chip, start=struct_starting[3], direction=180, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct4', Structure(chip, start=struct_starting[4], direction=90, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct5', Structure(chip, start=struct_starting[5], direction=90, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct6', Structure(chip, start=struct_starting[6], direction=90, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct7', Structure(chip, start=struct_starting[7], direction=90, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct8', Structure(chip, start=struct_starting[8], direction=0, defaults=defaults, layer = '0'))
#     setattr(chip, 'struct9', Structure(chip, start=struct_starting[9], direction=270, defaults=defaults, layer = '0'))


def cover_launchers(chip, chip_defaults, exclude = [], length = 500, width = 400, solid=0):
    """
    Cover the launchers with a square so that we can wirebond to the pads on the chip.
    """

    taper_length = 250
    taper_to_width = 2*50+20

    s = Structure(chip, start=chip.top_midpt, direction=270, defaults=chip_defaults)

    struct_starting = struct_startings(chip)


    if not(0 in exclude): #middle, top
        lo_left = (struct_starting[0][0], struct_starting[0][1] - width/2.)
        lo_right = (struct_starting[0][0] + (length-taper_length), struct_starting[0][1] -width/2.)
        tp_left = (struct_starting[0][0], struct_starting[0][1] +width/2.)
        tp_right = (struct_starting[0][0] + (length-taper_length), struct_starting[0][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = lo_right
        tp_left = tp_right
        midx = tp_left[0] + taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_right = (midx, midy + taper_to_width/2.)
        lo_right = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(1 in exclude): #middle, top
        lo_left = (struct_starting[1][0], struct_starting[1][1] - width/2.)
        lo_right = (struct_starting[1][0] + (length-taper_length), struct_starting[1][1] -width/2.)
        tp_left = (struct_starting[1][0], struct_starting[1][1] +width/2.)
        tp_right = (struct_starting[1][0] + (length-taper_length), struct_starting[1][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = lo_right
        tp_left = tp_right
        midx = tp_left[0] + taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_right = (midx, midy + taper_to_width/2.)
        lo_right = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(2 in exclude): #middle, top
        lo_left = (struct_starting[2][0], struct_starting[2][1] - width/2.)
        lo_right = (struct_starting[2][0] + (length-taper_length), struct_starting[2][1] -width/2.)
        tp_left = (struct_starting[2][0], struct_starting[2][1] +width/2.)
        tp_right = (struct_starting[2][0] + (length-taper_length), struct_starting[2][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = lo_right
        tp_left = tp_right
        midx = tp_left[0] + taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_right = (midx, midy + taper_to_width/2.)
        lo_right = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(3 in exclude): #Bottom, right
        lo_left = (struct_starting[3][0] -width/2., struct_starting[3][1] -(length-taper_length))
        lo_right = (struct_starting[3][0] + width/2., struct_starting[3][1] -(length-taper_length))
        tp_left = (struct_starting[3][0] -width/2., struct_starting[3][1])
        tp_right = (struct_starting[3][0] +width/2., struct_starting[3][1])

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        tp_left = lo_left
        tp_right = lo_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] - taper_length
        lo_left = (midx - taper_to_width/2., midy)
        lo_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(4 in exclude): #Bottom, right
        lo_left = (struct_starting[4][0] -width/2., struct_starting[4][1] -(length-taper_length))
        lo_right = (struct_starting[4][0] + width/2., struct_starting[4][1] -(length-taper_length))
        tp_left = (struct_starting[4][0] -width/2., struct_starting[4][1])
        tp_right = (struct_starting[4][0] +width/2., struct_starting[4][1])

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        tp_left = lo_left
        tp_right = lo_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] - taper_length
        lo_left = (midx - taper_to_width/2., midy)
        lo_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(5 in exclude): #Top, left
        lo_left = (struct_starting[5][0] -width/2., struct_starting[5][1] -(length-taper_length))
        lo_right = (struct_starting[5][0] + width/2., struct_starting[5][1] -(length-taper_length))
        tp_left = (struct_starting[5][0] -width/2., struct_starting[5][1])
        tp_right = (struct_starting[5][0] +width/2., struct_starting[5][1])

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        tp_left = lo_left
        tp_right = lo_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] - taper_length
        lo_left = (midx - taper_to_width/2., midy)
        lo_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))


    if not(6 in exclude): #Top, right
        lo_left = (struct_starting[6][0] - (length-taper_length), struct_starting[6][1] -width/2.)
        lo_right = (struct_starting[6][0], struct_starting[6][1] -width/2.)
        tp_left = (struct_starting[6][0] -(length-taper_length), struct_starting[6][1] +width/2.)
        tp_right = (struct_starting[6][0], struct_starting[6][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_right = lo_left
        tp_right = tp_left
        midx = tp_left[0] - taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_left = (midx, midy + taper_to_width/2.)
        lo_left = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(7 in exclude): #Top, right
        lo_left = (struct_starting[7][0] - (length-taper_length), struct_starting[7][1] -width/2.)
        lo_right = (struct_starting[7][0], struct_starting[7][1] -width/2.)
        tp_left = (struct_starting[7][0] -(length-taper_length), struct_starting[7][1] +width/2.)
        tp_right = (struct_starting[7][0], struct_starting[7][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_right = lo_left
        tp_right = tp_left
        midx = tp_left[0] - taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_left = (midx, midy + taper_to_width/2.)
        lo_left = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))



    if not(8 in exclude): #Top, right
        lo_left = (struct_starting[8][0] - (length-taper_length), struct_starting[8][1] -width/2.)
        lo_right = (struct_starting[8][0], struct_starting[8][1] -width/2.)
        tp_left = (struct_starting[8][0] -(length-taper_length), struct_starting[8][1] +width/2.)
        tp_right = (struct_starting[8][0], struct_starting[8][1] +width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_right = lo_left
        tp_right = tp_left
        midx = tp_left[0] - taper_length
        midy = (lo_left[1] + tp_left[1])/2.
        tp_left = (midx, midy + taper_to_width/2.)
        lo_left = (midx, midy - taper_to_width/2.)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(9 in exclude): #Right, middle
        lo_left = (struct_starting[9][0] -width/2., struct_starting[9][1])
        lo_right = (struct_starting[9][0] + width/2., struct_starting[9][1])
        tp_left = (struct_starting[9][0] -width/2., struct_starting[9][1]+(length-taper_length))
        tp_right = (struct_starting[9][0] +width/2., struct_starting[9][1]+(length-taper_length))

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = tp_left
        lo_right = tp_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] + taper_length
        tp_left = (midx - taper_to_width/2., midy)
        tp_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(10 in exclude): #Right, middle
        lo_left = (struct_starting[10][0] -width/2., struct_starting[10][1])
        lo_right = (struct_starting[10][0] + width/2., struct_starting[10][1])
        tp_left = (struct_starting[10][0] -width/2., struct_starting[10][1]+(length-taper_length))
        tp_right = (struct_starting[10][0] +width/2., struct_starting[10][1]+(length-taper_length))

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = tp_left
        lo_right = tp_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] + taper_length
        tp_left = (midx - taper_to_width/2., midy)
        tp_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))

    if not(11 in exclude): #Right, middle
        lo_left = (struct_starting[11][0] -width/2., struct_starting[11][1])
        lo_right = (struct_starting[11][0] + width/2., struct_starting[11][1])
        tp_left = (struct_starting[11][0] -width/2., struct_starting[11][1]+(length-taper_length))
        tp_right = (struct_starting[11][0] +width/2., struct_starting[11][1]+(length-taper_length))

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))

        lo_left = tp_left
        lo_right = tp_right
        midx = (lo_left[0] + lo_right[0])/2.
        midy = tp_left[1] + taper_length
        tp_left = (midx - taper_to_width/2., midy)
        tp_right = (midx + taper_to_width/2., midy)

        if solid:
            s.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
        else:
            s.append(sdxf.PolyLine([lo_left, tp_left, tp_right, lo_right, lo_left]))



def draw_chip_alignment_marks(solid,chip,chip_defaults):
    """
    Draw the alignment marks on the chip.
    """

    w = 80
    arml = 80
    CrossShapeAlignmentMarks(Structure(chip, start=(125,125), direction=90, defaults=chip_defaults), width = w,
            armlength = arml, solid = solid, layer = '0')
    CrossShapeAlignmentMarks(Structure(chip, start=(chip.size[0]-125, chip.size[1]-125), direction=90, defaults=chip_defaults),
            width = w, armlength = arml, solid = solid, layer = '0')
    CrossShapeAlignmentMarks(Structure(chip, start=(chip.size[0]-125, 125), direction=90, defaults=chip_defaults),
            width = w, armlength = arml, solid = solid, layer = '0')
    CrossShapeAlignmentMarks(Structure(chip, start=(125, chip.size[1]-125), direction=90, defaults=chip_defaults),
            width = w, armlength = arml, solid = solid, layer = '0')

def create_wirebond_border(chip, chip_default, solid, lo_left, lo_right, tp_left, tp_right):
    #
    structure = Structure(chip, start=chip.top_midpt, direction=270, defaults=chip_default)

    if solid:
        structure.append(sdxf.Solid([lo_left, lo_right, tp_right, tp_left]))
    else:
        structure.append(sdxf.PolyLine([lo_left, lo_right, tp_right, tp_left, lo_left]))


def draw_launchers(chip, chip_defaults, pinw, gapw, exclude = []):

    chipInit(chip, defaults=chip_defaults)

    for k in [0,1,2,3,4,5,6,7,8,9]:

        if k in exclude:
            pass
        else:
            Launcher(vars(vars()['chip'])['struct%d'%k], pinw = pinw, gapw = gapw, flipped=False, pad_length=250+50, taper_length=150, pad_to_length=350,)

def draw_launchers_10mm(chip, chip_defaults, pinw, gapw, exclude = []):

    chipInit_10mm(chip, defaults=chip_defaults)

    for k in [0,1,2,3,4,5,6,7,8,9]:

        if k in exclude:
            pass
        else:
            Launcher(vars(vars()['chip'])['struct%d'%k], pinw = pinw, gapw = gapw, flipped=False, pad_length=250+50, taper_length=150, pad_to_length=350,)

def draw_launchers_10by20mm(chip, chip_defaults, pinw, gapw, exclude = []):

    chipInit_10by20mm(chip, defaults=chip_defaults)

    for k in range(20):

        if k in exclude:
            pass
        else:
            Launcher(vars(vars()['chip'])['struct%d'%k], pinw = pinw, gapw = gapw, flipped=False, pad_length=250+50, taper_length=150, pad_to_length=350,)
def draw_launchers_10by20mmwide(chip, chip_defaults, pinw, gapw, exclude = []):

    chipInit_10by20mmwide(chip, defaults=chip_defaults)

    for k in range(20):

        if k in exclude:
            pass
        else:
            Launcher(vars(vars()['chip'])['struct%d'%k], pinw = pinw, gapw = gapw, flipped=False, pad_length=250+50, taper_length=150, pad_to_length=350,)