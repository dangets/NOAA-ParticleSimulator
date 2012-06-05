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
                    theta = math.atan2(ylen, xlen)

                    u = r * -math.sin(theta)
                    v = r * math.cos(theta)
                    w = 0.0

                    ofile.write("%5.2f %5.2f %5.2f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")


def genOutwardSpiral(ofile, numx, numy, numz, numt):
    ofile.write(" ".join([str(numx), str(numy), str(numz), str(numt)]))
    ofile.write("\n")

    cx = (numx - 1) * 0.5
    cy = (numy - 1) * 0.5
    cz = (numz - 1) * 0.5

    for t in range(numt):
        for z in range(numz):
            zlen = z - cz
            for y in range(numy):
                ylen = y - cy 
                for x in range(numx):
                    xlen = x - cx

                    u = xlen + ylen
                    v = ylen - xlen
                    w = 0.0

                    ofile.write("%5.2f %5.2f %5.2f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")


def genMiscCurve(ofile, numx, numy, numz, numt):
    ofile.write(" ".join([str(numx), str(numy), str(numz), str(numt)]))
    ofile.write("\n")

    cx = (numx - 1) * 0.5
    cy = (numy - 1) * 0.5
    cz = (numz - 1) * 0.5

    for t in range(numt):
        for z in range(numz):
            zlen = z - cz
            for y in range(numy):
                ylen = y - cy
                y2 = ylen * ylen
                for x in range(numx):
                    xlen = x - cx
                    x2 = xlen * xlen

                    u = x2 - y2
                    v = -2 * xlen * ylen
                    w = 0.0

                    ofile.write("%5.2f %5.2f %5.2f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")



if __name__ == "__main__":
    #genOutwardSpiral(sys.stdout, 64, 64, 16, 1)
    genMiscCurve(sys.stdout, 64, 64, 16, 1)

