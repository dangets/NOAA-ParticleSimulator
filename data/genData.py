import ctypes
import math
import sys

def genDataSphere(ofile, numx, numy, numz, numt):
    ofile.write(" ".join([str(numx), str(numy), str(numz), str(numt)]))
    ofile.write("\n")

    cx = (numx - 1) * 0.5
    cy = (numy - 1) * 0.5
    cz = (numz - 1) * 0.5

    for t in range(numt):
        for z in range(numz):
            zlen = z - cz
            z2 = zlen * zlen
            for y in range(numy):
                ylen = y - cy 
                y2 = ylen * ylen
                for x in range(numx):
                    xlen = x - cx
                    x2 = xlen * xlen

                    r = math.sqrt(x2 + y2 + z2)

                    r *= 0.5

                    if xlen < 0:
                        u = -r
                    else:
                        u = r
                    if ylen < 0:
                        v = -r
                    else:
                        v = r

                    w = 0.0

                    ofile.write("%5.2f %5.2f %5.2f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")


if __name__ == "__main__":
    genDataSphere(sys.stdout, 64, 64, 16, 1)
