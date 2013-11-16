import ctypes
import math
import sys


def genCylinder(ofile, numx, numy, numz, numt):
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

                    u = -ylen
                    v = xlen
                    w = 0.0

                    ofile.write("%f %f %f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")


def genCylinderNormalized(ofile, numx, numy, numz, numt):
    ofile.write(" ".join([str(numx), str(numy), str(numz), str(numt)]))
    ofile.write("\n")

    cx = (numx - 1) * 0.5
    cy = (numy - 1) * 0.5
    cz = (numz - 1) * 0.5

    dx = 2.0 / numx
    dy = 2.0 / numy

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

                    u = -ylen * dy
                    v = xlen * dx
                    w = 0.0

                    ofile.write("%f %f %f " % (u, v, w))
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

                    ofile.write("%f %f %f " % (u, v, w))
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

                    ofile.write("%f %f %f " % (u, v, w))
                ofile.write("\n")
            ofile.write("\n")



if __name__ == "__main__":
    #genCylinder(sys.stdout, 64, 64, 16, 1)
    genCylinderNormalized(sys.stdout, 64, 64, 16, 1)
    #genOutwardSpiral(sys.stdout, 64, 64, 16, 1)
    #genMiscCurve(sys.stdout, 64, 64, 16, 1)

