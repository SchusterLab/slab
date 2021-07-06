"""
qsystem0.py
"""

import re
import time

import numpy as np
from pynq import Overlay, allocate, Xlnk
from pynq.lib import AxiGPIO
try:
    import xrfdc
    import xrfclk
except Exception as e:
    xrfdc = None
    xrfclk = None
#ENDTRY

# Function to parse program.
def parse_prog(file="prog.asm",outfmt="bin"):
    
    # Output structure.
    outProg = {}

    # Instructions.
    instList = {}

    # I-type.
    instList['pushi']   = {'bin':'00010000'}
    instList['popi']    = {'bin':'00010001'}
    instList['mathi']   = {'bin':'00010010'}
    instList['seti']    = {'bin':'00010011'}
    instList['synci']   = {'bin':'00010100'}
    instList['waiti']   = {'bin':'00010101'}
    instList['bitwi']   = {'bin':'00010110'}
    instList['memri']   = {'bin':'00010111'}
    instList['memwi']   = {'bin':'00011000'}
    instList['regwi']   = {'bin':'00011001'}

    # J-type.
    instList['loopnz']  = {'bin':'00110000'}
    instList['condj']   = {'bin':'00110001'}
    instList['end']     = {'bin':'00111111'}

    # R-type.
    instList['math']    = {'bin':'01010000'}
    instList['set']     = {'bin':'01010001'}
    instList['sync']    = {'bin':'01010010'}
    instList['read']    = {'bin':'01010011'}
    instList['wait']    = {'bin':'01010100'}
    instList['bitw']    = {'bin':'01010101'}
    instList['memr']    = {'bin':'01010110'}
    instList['memw']    = {'bin':'01010111'}


    # Structures for symbols and program.
    progList = {}
    symbList = {}

    ##############################
    ### Read program from file ###
    ##############################
    fd = open(file,"r")
    addr = 0
    for line in fd:
        # Match comments.
        m = re.search("^\s*//", line)
        
        # If there is a match.
        if m:
            #print(line)
            a = 1
        
        else:
            # Match instructions.
            jump_re = "^((.+):)?"
            inst_re_I = "pushi|popi|mathi|seti|synci|waiti|bitwi|memri|memwi|regwi|";
            inst_re_J = "loopnz|condj|end|";
            inst_re_R = "math|set|sync|read|wait|bitw|memr|memw";
            inst_re = "\s*(" + inst_re_I + inst_re_J + inst_re_R + ")\s+(.+);";
            comp_re = jump_re + inst_re
            m = re.search(comp_re, line, flags = re.MULTILINE)
    
            # If there is a match.
            if m:
                # Tagged instruction for jump.
                if m.group(2):
                    symb = m.group(2)
                    inst = m.group(3)
                    args = m.group(4)
            
                    # Add symbol to symbList.
                    symbList[symb] = addr;
            
                    # Add instruction to progList.
                    progList[addr] = {'inst':inst,'args':args}
            
                    # Increment address.
                    addr = addr + 1
            
                # Normal instruction.
                else:
                    inst = m.group(3)
                    args = m.group(4)
            
                    # Add instruction to progList.
                    progList[addr] = {'inst':inst,'args':args}

                    # Increment address.
                    addr = addr + 1
                
            # Check special case of "end" instruction.
            else:
                m = re.search("\s*(end);",line)

                # If there is a match.
                if m:
                    # Add instruction to progList.
                    progList[addr] = {'inst':'end','args':''}
                
                    # Increment address.
                    addr = addr + 1

    #########################
    ### Support functions ###
    #########################
    def unsigned2bin(strin,bits=8):
        maxv = 2**bits - 1
    
        # Check if hex string.
        m = re.search("^0x", strin, flags = re.MULTILINE)
        if m:
            dec = int(strin, 16)
        else:
            dec = int(strin, 10)
        
        # Check max.
        if dec > maxv:
            print("Error: number %d is bigger than %d" %(dec,maxv))
            return None
    
        # Convert to binary.
        fmt = "{0:0" + str(bits) + "b}"
        binv = fmt.format(dec)
        
        return binv
    
    def integer2bin(strin,bits=8):
        minv = -2**(bits-1)
        maxv = 2**(bits-1) - 1
    
        # Check if hex string.
        m = re.search("^0x", strin, flags = re.MULTILINE)
        if m:
            # Special case for hex number.
            dec = int(strin, 16)
            
            # Convert to binary.
            fmt = "{0:0" + str(bits) + "b}"
            binv = fmt.format(dec)
        
            return binv
        else:
            dec = int(strin, 10)
        
        # Check max.
        if dec < minv:
            print("Error: number %d is smaller than %d" %(dec,minv))
            return None
        
        # Check max.
        if dec > maxv:
            print("Error: number %d is bigger than %d" %(dec,maxv))
            return None
    
        # Check if number is negative.
        if dec < 0:
            dec = dec + 2**bits
            
        # Convert to binary.
        fmt = "{0:0" + str(bits) + "b}"
        binv = fmt.format(dec)
        
        return binv

    def op2bin(op):
        if op == "0":
            return "0000"
        elif op == ">":
            return "0000"
        elif op == ">=":
            return "0001"
        elif op == "<":
            return "0010"
        elif op == "<=":
            return "0011"
        elif op == "==":
            return "0100"
        elif op == "!=":
            return "0101"
        elif op == "+":
            return "1000"
        elif op == "-":
            return "1001"
        elif op == "*":
            return "1010"
        elif op == "&":
            return "0000"
        elif op == "|":
            return "0001"
        elif op == "^":
            return "0010"
        elif op == "~":
            return "0011"
        elif op == "<<":
            return "0100"
        elif op == ">>":
            return "0101"
        else:
            print("Error: operation \"%s\" not recognized" % op)
            return "1111"

    ######################################
    ### First pass: parse instructions ###
    ######################################
    for e in progList:
        inst = progList[e]['inst']
        args = progList[e]['args']
        
        # I-type: three registers and an immediate value.
        # I-type:<inst>:page:channel:oper:ra:rb:rc:imm
    
        # pushi p, $ra, $rb, imm
        if inst == 'pushi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*,\s*(\-?\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
                imm  = m.group(4)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:pushi:" + page + ":0:0:" + rb + ":" + ra + ":0:" + imm
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))
            
        # popi p, $r
        elif inst == 'popi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)

                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:popi:" + page + ":0:0:" + r + ":0:0:0"
            
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))
            
        # mathi p, $ra, $rb oper imm
        if inst == 'mathi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*([\+\-\*])\s*(0?x?\-?[0-9a-fA-F]+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
                oper = m.group(4)
                imm  = m.group(5)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:mathi:" + page + ":0:" + oper + ":" + ra + ":" + rb + ":0:" + imm
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))                
            
        # seti ch, p, $r, t
        if inst == 'seti':
            comp_re = "\s*(\d+)\s*,\s*(\d+)\s*,\s*\$(\d+)\s*,\s*(\-?\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                ch   = m.group(1)
                page = m.group(2)
                ra   = m.group(3)
                t    = m.group(4)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:seti:" + page + ":" + ch + ":0:0:" + ra + ":0:" + t
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))         
            
        # synci t
        if inst == 'synci':
            comp_re = "\s*(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                t    = m.group(1)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:synci:0:0:0:0:0:0:" + t
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))      
            
        # waiti ch, t
        if inst == 'waiti':
            comp_re = "\s*(\d+)\s*,\s*(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                ch   = m.group(1)
                t    = m.group(2)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:waiti:0:" + ch + ":0:0:0:0:" + t
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))  
                
        # bitwi p, $ra, $rb oper imm
        if inst == 'bitwi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*([&|<>^]+)\s*(0?x?\-?[0-9a-fA-F]+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
                oper = m.group(4)
                imm  = m.group(5)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:bitwi:" + page + ":0:" + oper + ":" + ra + ":" + rb + ":0:" + imm
        
            # bitwi p, $ra, ~imm
            else:                         
                comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*~\s*(0?x?\-?[0-9a-fA-F]+)";
                m = re.search(comp_re, args)
        
                # If there is a match.
                if m:
                    page = m.group(1)
                    ra   = m.group(2)
                    oper = "~"
                    imm  = m.group(3)
            
                    # Add entry into structure.
                    progList[e]['inst_parse'] = "I-type:bitwi:" + page + ":0:" + oper + ":" + ra + ":0:0:" + imm
        
                # Error: bad instruction format.
                else:
                    print("Error: bad format on instruction @%d: %s" %(e,inst))              

        # memri p, $r, imm
        if inst == 'memri':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*(0?x?\-?[0-9a-fA-F]+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)
                imm  = m.group(3)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:memri:" + page + ":0:0:" + r + ":0:0:" + imm
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))        
                
        # memwi p, $r, imm
        if inst == 'memwi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*(0?x?\-?[0-9a-fA-F]+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)
                imm  = m.group(3)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:memwi:" + page + ":0:0:0:0:" + r + ":" + imm
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))                  
            
        # regwi p, $r, imm
        if inst == 'regwi':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*(0?x?\-?[0-9a-fA-F]+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)
                imm  = m.group(3)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "I-type:regwi:" + page + ":0:0:" + r + ":0:0:" + imm
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))      
                
        # J-type: three registers and an address for jump.
        # J-type:<inst>:page:oper:ra:rb:rc:addr                
                
        # loopnz p, $r, @label
        if inst == 'loopnz':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\@(.+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page  = m.group(1)
                oper  = "+"
                r     = m.group(2)
                label = m.group(3)

                # Resolve symbol.
                if label in symbList:
                    label_addr = symbList[label]
                else:
                    print("Error: could not resolve symbol %s on instruction @%d: %s %s" %(label,e,inst,args))
            
                # Add entry into structure.
                regs = r + ":" + r + ":0:" + str(label_addr)
                progList[e]['inst_parse'] = "J-type:loopnz:" + page + ":" + oper + ":" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))    
                
        # condj p, $ra op $rb, @label
        if inst == 'condj':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*([<>=!]+)\s*\$(\d+)\s*,\s*\@(.+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page  = m.group(1)
                ra    = m.group(2)
                oper  = m.group(3)
                rb    = m.group(4)
                label = m.group(5)

                # Resolve symbol.
                if label in symbList:
                    label_addr = symbList[label]
                else:
                    print("Error: could not resolve symbol %s on instruction @%d: %s %s" %(label,e,inst,args))
            
                # Add entry into structure.
                regs = ra + ":" + rb + ":" + str(label_addr)
                progList[e]['inst_parse'] = "J-type:condj:" + page + ":" + oper + ":0:" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))                 
            
        # end
        if inst == 'end':       
            # Add entry into structure.
            progList[e]['inst_parse'] = "J-type:end:0:0:0:0:0:0"
        
        
        # R-type: 8 registers, 7 for reading and 1 for writing.
        # R-type:<inst>:page:channel:oper:ra:rb:rc:rd:re:rf:rg:rh            
        
        # math p, $ra, $rb oper $rc
        if inst == 'math':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*([\+\-\*])\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
                oper = m.group(4)
                rc   = m.group(5)
            
                # Add entry into structure.
                regs = ra + ":" + rb + ":" + rc + ":0:0:0:0:0"
                progList[e]['inst_parse'] = "R-type:math:" + page + ":0:" + oper + ":" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))       
            
        # set ch, p, $ra, $rb, $rc, $rd, $re, $rt
        if inst == 'set':
            regs = "\s*\$(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)"
            comp_re = "\s*(\d+)\s*,\s*(\d+)\s*," + regs;
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                ch   = m.group(1)
                page = m.group(2)
                ra   = m.group(3)
                rb   = m.group(4)
                rc   = m.group(5)
                rd   = m.group(6)
                ree  = m.group(7)
                rt   = m.group(8)
            
                # Add entry into structure.
                regs = ra + ":" + rt + ":" + rb + ":" + rc + ":" + rd + ":" + ree + ":0"
                progList[e]['inst_parse'] = "R-type:set:" + page + ":" + ch + ":0:0:" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))  
            
        # sync p, $r
        if inst == 'sync':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "R-type:sync:" + page + ":0:0:0:0:" + r + ":0:0:0:0:0"
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))    
            
        # read p, $r
        if inst == 'read':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                r    = m.group(2)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "R-type:read:" + page + ":0:0:" + r + ":0:0:0:0:0:0:0"
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))    
            
        # wait ch, p, $r
        if inst == 'wait':
            comp_re = "\s*(\d+)\s*,\s*(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                ch   = m.group(1)
                page = m.group(2)
                r    = m.group(3)
            
                # Add entry into structure.
                progList[e]['inst_parse'] = "R-type:wait:" + page + ":" + ch + ":0:0:0:" + r + ":0:0:0:0:0"
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))        
                
        # bitw p, $ra, $rb oper $rc
        if inst == 'bitw':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)\s*([&|<>^]+)\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
                oper = m.group(4)
                rc   = m.group(5)
            
                # Add entry into structure.
                regs = ra + ":" + rb + ":" + rc + ":0:0:0:0:0"
                progList[e]['inst_parse'] = "R-type:bitw:" + page + ":0:" + oper + ":" + regs
        
            # bitw p, $ra, ~$rb
            else:                              
                comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*~\s*\$(\d+)";
                m = re.search(comp_re, args)
        
                # If there is a match.
                if m:
                    page = m.group(1)
                    ra   = m.group(2)
                    rb   = m.group(3)
                    oper = "~"
            
                    # Add entry into structure.
                    regs = ra + ":0:" + ":" + rb + ":0:0:0:0:0"
                    progList[e]['inst_parse'] = "R-type:bitw:" + page + ":0:" + oper + ":" + regs
        
                # Error: bad instruction format.
                else:
                    print("Error: bad format on instruction @%d: %s" %(e,inst))                 
                
        # memr p, $ra, $rb
        if inst == 'memr':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
            
                # Add entry into structure.
                regs = ra + ":" + rb + ":0:0:0:0:0:0"
                progList[e]['inst_parse'] = "R-type:memr:" + page + ":0:0:" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst)) 
                
        # memw p, $ra, $rb
        if inst == 'memw':
            comp_re = "\s*(\d+)\s*,\s*\$(\d+)\s*,\s*\$(\d+)";
            m = re.search(comp_re, args)
        
            # If there is a match.
            if m:
                page = m.group(1)
                ra   = m.group(2)
                rb   = m.group(3)
            
                # Add entry into structure.
                regs = rb + ":" + ra + ":0:0:0:0:0"
                progList[e]['inst_parse'] = "R-type:memw:" + page + ":0:0:0:" + regs
        
            # Error: bad instruction format.
            else:
                print("Error: bad format on instruction @%d: %s" %(e,inst))                 
    
    ######################################
    ### Second pass: convert to binary ###
    ######################################
    for e in progList:
        inst = progList[e]['inst_parse']
        spl = inst.split(":")
    
        # I-type
        if spl[0] == "I-type":
            # Instruction.
            if spl[1] in instList:            
                inst_bin = instList[spl[1]]['bin']
            else:
                print("Error: instruction %s not found on instraction list" % spl[1])
            
            # page.
            page = unsigned2bin(spl[2],3)
            
            # channel
            ch = unsigned2bin(spl[3],3)
            
            # oper
            oper = op2bin(spl[4])
        
            # Registers.
            ra   = unsigned2bin(spl[5],5)
            rb   = unsigned2bin(spl[6],5)
            rc   = unsigned2bin(spl[7],5)
            
            # Zeros
            z15 = unsigned2bin("0",15)
        
            # Immediate.
            imm  = integer2bin(spl[8],16)
        
            # Machine code (bin/hex).
            code = inst_bin + page + ch + oper + ra + rb + rc + z15 + imm
            code_h = "{:016x}".format(int(code,2))
        
            # Write values back into hash.
            progList[e]['inst_bin'] = code
            progList[e]['inst_hex'] = code_h
        
        elif (spl[0] == "J-type"):
            # Instruction.
            if spl[1] in instList:            
                inst_bin = instList[spl[1]]['bin']
            else:
                print("Error: instruction %s not found on instraction list" % spl[1])  
            
            # Page.
            page = unsigned2bin(spl[2],3)
            
            # Zeros.
            z3 = unsigned2bin("0",3)
            
            #oper
            oper = op2bin(spl[3])
        
            # Registers.
            ra = unsigned2bin(spl[4],5)
            rb = unsigned2bin(spl[5],5)
            rc = unsigned2bin(spl[6],5)
            
            # Zeros.
            z15 = unsigned2bin("0",15)
        
            # Address.
            jmp_addr = unsigned2bin(spl[7],16)
        
            # Machine code (bin/hex).
            code = inst_bin + page + z3 + oper + ra + rb + rc + z15 + jmp_addr
            code_h = "{:016x}".format(int(code,2))
        
            # Write values back into hash.
            progList[e]['inst_bin'] = code
            progList[e]['inst_hex'] = code_h 
              
        elif (spl[0] == "R-type"):
            # Instruction.
            if spl[1] in instList:            
                inst_bin = instList[spl[1]]['bin']
            else:
                print("Error: instruction \"%s\" not found on instraction list" % spl[1])        
        
            # Page.
            page = unsigned2bin(spl[2],3)
            
            # Channel
            ch = unsigned2bin(spl[3],3)
            
            # Oper
            oper = op2bin(spl[4])
        
            # Registers.
            ra   = unsigned2bin(spl[5],5)
            rb   = unsigned2bin(spl[6],5)
            rc   = unsigned2bin(spl[7],5)
            rd   = unsigned2bin(spl[8],5)
            ree   = unsigned2bin(spl[9],5)  
            rf   = unsigned2bin(spl[10],5)
            rg   = unsigned2bin(spl[11],5)
            rh   = unsigned2bin(spl[12],5)            
            
            # Zeros.
            z6 = unsigned2bin("0",6)
        
            # Machine code (bin/hex).
            code = inst_bin + page + ch + oper + ra + rb + rc + rd + ree + rf + rg + rh + z6
            code_h = "{:016x}".format(int(code,2))
        
            # Write values back into hash.
            progList[e]['inst_bin'] = code
            progList[e]['inst_hex'] = code_h            
        
        else:
            print("Error: bad type on instruction @%d: %s" %(e,inst))
            
    ####################
    ### Write output ###
    ####################
    # Binary format.
    if outfmt == "bin":
        for e in progList:
            outProg[e] = progList[e]['inst_bin']
            
    # Hex format.
    elif outfmt == "hex":
        for e in progList:
            out = progList[e]['inst_hex'] + " -> " + progList[e]['inst'] + " " + progList[e]['args']
            outProg[e] = out
    
    else:
        print("Error: \"%s\" is not a recognized output format" % outfmt)
    
    # Return program list.
    return outProg

# Support functions.
def gauss(mu=0,si=0,length=100,maxv=30000):
    x = np.arange(0,length)
    y = 1/(2*np.pi*si**2)*np.exp(-(x-mu)**2/si**2)
    y = y/np.max(y)*maxv
    return y

def triang(length=100,maxv=30000):
    y1 = np.arange(0,length/2)
    y2 = np.flip(y1,0)
    y = np.concatenate((y1,y2))
    y = y/np.max(y)*maxv
    return y

def freq2reg(fs,f,B=16):
    df = 2**B/fs
    f_i = f*df
    return int(f_i)

def reg2freq(fs, r, B=16):
    return r*fs/2**B

# Some support functions
def format_buffer(buff):
    # Format: 
    # -> lower 16 bits: I value.
    # -> higher 16 bits: Q value.
    data = buff
    dataI = data & 0xFFFF
    dataI = dataI.astype(np.int16)
    dataQ = data >> 16
    dataQ = dataQ.astype(np.int16)
    
    return dataI,dataQ

class SocIp:
    REGISTERS = {}    
    
    def __init__(self, ip, **kwargs):
        self.ip = ip
        
    def write(self, offset, s):
        self.ip.write(offset, s)
        
    def read(self, offset):
        return self.ip.read(offset)
    
    def __setattr__(self, a ,v):
        if a in self.__class__.REGISTERS:
            self.ip.write(4*self.__class__.REGISTERS[a], v)
        else:
            return super().__setattr__(a,v)
    
    def __getattr__(self, a):
        if a in self.__class__.REGISTERS:
            return self.ip.read(4*self.__class__.REGISTERS[a])
        else:
            return super().__getattr__(a)           
        
class AxisSignalGenV3(SocIp):
    # AXIS Table Registers.
    # START_ADDR_REG
    #
    # WE_REG
    # * 0 : disable writes.
    # * 1 : enable writes.
    #
    REGISTERS = {'start_addr_reg':0, 'we_reg':1}
    
    # Generics
    N = 12
    NDDS = 16
    
    # Maximum number of samples
    MAX_LENGTH = 2**N*NDDS
    
    def __init__(self, ip, axi_dma, dds_mr_switch, axis_switch, channel, name, **kwargs):
        # Init IP.
        super().__init__(ip)
        
        # Default registers.
        self.start_addr_reg=0
        self.we_reg=0
        
        # dma
        self.dma = axi_dma
        
        # Real/imaginary selection switch.
        self.iq_switch = AxisDdsMrSwitch(dds_mr_switch)
        
        # switch
        self.switch = AxisSwitch(axis_switch)
        
        # Channel.
        self.ch = channel
        
        # Name.
        self.name = name
        
    # Load waveforms.
    def load(self, buff_in,addr=0):
        # Route switch to channel.
        self.switch.sel(self.ch)
        
        time.sleep(0.1)
        
        # Define buffer.
        #self.buff = Xlnk().cma_array(shape=(len(buff_in)), dtype=np.int16)
        self.buff = allocate(shape=(len(buff_in)), dtype=np.int16)
        
        ###################
        ### Load I data ###
        ###################
        np.copyto(self.buff,buff_in)

        # Enable writes.
        self.wr_enable(addr)

        # DMA data.
        self.dma.sendchannel.transfer(self.buff)
        self.dma.sendchannel.wait()

        # Disable writes.
        self.wr_disable()        
        
    def wr_enable(self,addr=0):
        self.start_addr_reg = addr
        self.we_reg = 1
        
    def wr_disable(self):
        self.we_reg = 0
        
class AxisSwitch(SocIp):
    REGISTERS = {'ctrl': 0x0, 'mix_mux': 0x040}
    
    # Number of master interfaces.
    NMI = 4
    
    def __init__(self, switch, **kwargs):
        super().__init__(switch)
        self.switch = self.ip
        
        # Init axis_switch.
        self.ctrl = 0
        self.disable_ports()
            
    def disable_ports(self):
        for ii in range(self.NMI):
            offset = self.REGISTERS['mix_mux'] + 4*ii
            self.write(offset,0x80000000)
        
    def sel(self,ch):
        # Disable register update.
        self.ctrl = 0

        # Disable all MI ports.
        self.disable_ports()
        
        # MI[0] -> SI[ch]
        offset = self.REGISTERS['mix_mux'] + 4*ch
        self.write(offset,0x00000000)

        # Enable register update.
        self.ctrl = 2             
        
class AxisDdsMrSwitch(SocIp):
    # AXIS DDS MR SWITCH registers.
    # DDS_REAL_IMAG_REG
    # * 0 : real part.
    # * 1 : imaginary part.
    #
    REGISTERS = {'dds_real_imag' : 0}
    
    def __init__(self, ip):
        # Initialize ip
        super().__init__(ip)
        
        # Default registers.
        # dds_real_imag = 0  : take real part.
        self.dds_real_imag = 0
        
    def real(self):
        self.dds_real_imag = 0
        
    def imag(self):
        self.dds_real_imag = 1
        
class AxisTProc64x8(SocIp):
    # AXIS tProcessor registers.
    # START_SRC_REG
    # * 0 : internal start.
    # * 1 : external start.
    #
    # START_REG
    # * 0 : stop.
    # * 1 : start.
    #
    # MEM_MODE_REG
    # * 0 : AXIS Read (from memory to m0_axis)
    # * 1 : AXIS Write (from s0_axis to memory)
    #
    # MEM_START_REG
    # * 0 : Stop.
    # * 1 : Execute operation (AXIS)
    #
    # MEM_ADDR_REG : starting memory address for AXIS read/write mode.
    #
    # MEM_LEN_REG : number of samples to be transferred in AXIS read/write mode.
    #
    REGISTERS = {'start_src_reg' : 0, 
                 'start_reg' : 1, 
                 'mem_mode_reg' : 2, 
                 'mem_start_reg' : 3, 
                 'mem_addr_reg' : 4, 
                 'mem_len_reg' : 5}
    
    # Generics.
    DMEM_N = 10
    PMEM_N = 16
    AXI_ADDR_WIDTH = DMEM_N + 1
    
    def __init__(self, ip, mem, axi_dma):
        # Initialize ip
        super().__init__(ip)
        
        # Program memory.
        self.mem = mem
        
        # Default registers.
        # start_src_reg = 0   : internal start.
        # start_reg     = 0   : stopped.
        # mem_mode_reg  = 0   : axis read.
        # mem_start_reg = 0   : axis operation stopped.
        # mem_addr_reg  = 0   : start address = 0.
        # mem_len_reg   = 100 : default length.
        self.start_src_reg = 0
        self.start_reg     = 0
        self.mem_mode_reg  = 0
        self.mem_start_reg = 0
        self.mem_addr_reg  = 0
        self.mem_len_reg   = 100
        
        # dma
        self.dma = axi_dma 
        
    def start_src(self,src=0):
        self.start_src_reg = src
        
    def start(self):
        self.start_reg = 1
        
    def stop(self):
        self.start_reg = 0
        
    def load_asm_program(self, prog):
        """
        prog -- the ASM_program to load 
        """
        for ii,inst in enumerate(prog.compile()):

            dec_low = int(inst & 0xffffffff)
            dec_high = int(inst >> 32)
            #print(hex(inst), dec_low, dec_high,type(dec_low), type(dec_high))
            self.mem.write(offset=8*ii,value=dec_low)
            self.mem.write(offset=4*(2*ii+1),value=dec_high)

    def load_program(self,prog="prog.asm",fmt="asm"):
        # Binary file format.
        if fmt == "bin":
            # Read binary file from disk.
            fd = open(prog,"r")
            
            # Write memory.
            addr = 0
            for line in fd:
                line.strip("\r\n")
                dec = int(line,2)
                dec_low = dec & 0xffffffff
                dec_high = dec >> 32
                self.mem.write(offset=addr,value=dec_low)
                addr = addr + 4
                self.mem.write(offset=addr,value=dec_high)
                addr = addr + 4                
                
        # Asm file.
        elif fmt == "asm":
            # Compile program.
            progList = parse_prog(prog)
        
            # Load Program Memory.
            addr = 0
            for e in progList:
                dec = int(progList[e],2)
                #print ("@" + str(addr) + ": " + str(dec))
                dec_low = dec & 0xffffffff
                dec_high = dec >> 32
                self.mem.write(offset=addr,value=dec_low)
                addr = addr + 4
                self.mem.write(offset=addr,value=dec_high)
                addr = addr + 4   
                
    def single_read(self, addr):
        # Address should be translated to uppder map.
        addr_temp = 4*addr + 2**self.DMEM_N
        #print(addr_temp)
            
        # Read data.
        data = self.ip.read(offset=addr_temp)
            
        return data
    
    def single_write(self, addr=1024, data=0):
        # Address should be translated to uppder map.
        addr_temp = 4*addr + 2**self.DMEM_N
            
        # Write data.
        self.ip.write(offset=addr_temp,value=data)
        
    def load_dmem(self, buff_in, addr=0):
        # Length.
        length = len(buff_in)
        
        # Configure dmem arbiter.
        self.mem_mode_reg = 1
        self.mem_addr_reg = addr
        self.mem_len_reg = length
        
        # Define buffer.
        self.buff = Xlnk().cma_array(shape=(length), dtype=np.int16)
        
        # Copy buffer.
        np.copyto(self.buff,buff_in)

        # Start operation on block.
        self.mem_start_reg = 1

        # DMA data.
        self.dma.sendchannel.transfer(self.buff)
        self.dma.sendchannel.wait()

        # Set block back to single mode.
        self.mem_start_reg = 0
        
    def read_dmem(self, addr=0, length=100):
        # Configure dmem arbiter.
        self.mem_mode_reg = 0
        self.mem_addr_reg = addr
        self.mem_len_reg = length
        
        # Define buffer.
        buff = Xlnk().cma_array(shape=(length), dtype=np.int16)
        
        # Start operation on block.
        self.mem_start_reg = 1

        # DMA data.
        self.dma.recvchannel.transfer(buff)
        self.dma.recvchannel.wait()

        # Set block back to single mode.
        self.mem_start_reg = 0
        
        return buff
    
class MrBufferEt(SocIp):
    # Registers.
    # DW_CAPTURE_REG
    # * 0 : Capture disabled.
    # * 1 : Capture enabled (capture started by external trigger).
    #
    # DR_START_REG
    # * 0 : don't send.
    # * 1 : start sending data.
    #
    # DW_CAPTURE_REG needs to be de-asserted and asserted again to allow a new capture.
    # DR_START_REG needs to be de-assereted and asserted again to allow a new transfer.
    #
    REGISTERS = {'dw_capture_reg':0, 'dr_start_reg':1}
    
    # Generics
    N = 14
    Nm = 8
        
    # Maximum number of samples
    MAX_LENGTH = 2**N * Nm   
    
    def __init__(self, ip, axi_dma, **kwargs):
        # Init IP.
        super().__init__(ip)
        
        # Default registers.
        self.dw_capture_reg=0
        self.dr_start_reg=0
        
        # dma
        self.dma = axi_dma
        
    def transfer(self,buff):       
        # Start send data mode.
        self.dr_start_reg = 1
        
        # DMA data.
        self.dma.recvchannel.transfer(buff)
        self.dma.recvchannel.wait()

        # Stop send data mode.
        self.dr_start_reg = 0  
        
    def enable(self):
        self.dw_capture_reg = 1
        
    def disable(self):
        self.dw_capture_reg = 0
        
class AxisAvgBuffer(SocIp):
    # Registers.
    # AVG_START_REG
    # * 0 : Averager Disabled.
    # * 1 : Averager Enabled (started by external trigger).
    #
    # AVG_ADDR_REG : start address to write results.
    #
    # AVG_LEN_REG : number of samples to be added.
    #
    # AVG_DR_START_REG
    # * 0 : do not send any data.
    # * 1 : send data using m0_axis.
    #
    # AVG_DR_ADDR_REG : start address to read data.
    #
    # AVG_DR_LEN_REG : number of samples to be read.
    #
    # BUF_START_REG
    # * 0 : Buffer Disabled.
    # * 1 : Buffer Enabled (started by external trigger).
    #
    # BUF_ADDR_REG : start address to write results.
    #
    # BUF_LEN_REG : number of samples to be buffered.
    #
    # BUF_DR_START_REG
    # * 0 : do not send any data.
    # * 1 : send data using m1_axis.
    #
    # BUF_DR_ADDR_REG : start address to read data.
    #
    # BUF_DR_LEN_REG : number of samples to be read.    
    #
    REGISTERS = {'avg_start_reg'    : 0, 
                 'avg_addr_reg'     : 1,
                 'avg_len_reg'      : 2,
                 'avg_dr_start_reg' : 3,
                 'avg_dr_addr_reg'  : 4,
                 'avg_dr_len_reg'   : 5,
                 'buf_start_reg'    : 6, 
                 'buf_addr_reg'     : 7,
                 'buf_len_reg'      : 8,
                 'buf_dr_start_reg' : 9,
                 'buf_dr_addr_reg'  : 10,
                 'buf_dr_len_reg'   : 11}
    
    # Generics
    B = 16
    N_AVG = 14
    N_BUF = 16
        
    # Maximum number of samples
    AVG_MAX_LENGTH = 2**N_AVG  
    BUF_MAX_LENGTH = 2**N_BUF
    
    def __init__(self, ip, axi_dma_avg, axi_dma_buf, **kwargs):
        # Init IP.
        super().__init__(ip)
        
        # Default registers.
        self.avg_start_reg    = 0
        self.avg_dr_start_reg = 0
        self.buf_start_reg    = 0
        self.buf_dr_start_reg = 0        
        
        # dmas
        self.dma_avg = axi_dma_avg
        self.dma_buf = axi_dma_buf

    def config(self,address=0,length=100):
        # Configure averaging and buffering to the same address and length.
        self.config_avg(address=address,length=length)
        self.config_buf(address=address,length=length)
        
    def enable(self):
        # Enable both averager and buffer.
        self.enable_avg()
        self.enable_buf()
        
    def config_avg(self,address=0,length=100):
        # Disable averaging.
        self.disable_avg()
        
        # Set registers.
        self.avg_addr_reg = address
        self.avg_len_reg = length
        
    def transfer_avg(self,buff,address=0,length=100):
        # Set averager data reader address and length.
        self.avg_dr_addr_reg = address
        self.avg_dr_len_reg = length
        
        # Start send data mode.
        self.avg_dr_start_reg = 1
        
        # DMA data.
        self.dma_avg.recvchannel.transfer(buff)
        self.dma_avg.recvchannel.wait()

        # Stop send data mode.
        self.avg_dr_start_reg = 0
        
        # Format: 
        # -> lower 32 bits: I value.
        # -> higher 32 bits: Q value.
        data = buff
        dataI = data & 0xFFFFFFFF
        dataI = dataI.astype(np.int32)
        dataQ = data >> 32
        dataQ = dataQ.astype(np.int32)
    
        return dataI,dataQ        
        
    def enable_avg(self):
        self.avg_start_reg = 1
        
    def disable_avg(self):
        self.avg_start_reg = 0    
        
    def config_buf(self,address=0,length=100):
        # Disable buffering.
        self.disable_buf()
        
        # Set registers.
        self.buf_addr_reg = address
        self.buf_len_reg = length    
        
    def transfer_buf(self,buff,address=0,length=100):
        # Set buffer data reader address and length.
        self.buf_dr_addr_reg = address
        self.buf_dr_len_reg = length
        
        # Start send data mode.
        self.buf_dr_start_reg = 1
        
        # DMA data.
        self.dma_buf.recvchannel.transfer(buff)
        self.dma_buf.recvchannel.wait()

        # Stop send data mode.
        self.buf_dr_start_reg = 0
        
        # Format: 
        # -> lower 16 bits: I value.
        # -> higher 16 bits: Q value.
        data = buff
        dataI = data & 0xFFFF
        dataI = dataI.astype(np.int16)
        dataQ = data >> 16
        dataQ = dataQ.astype(np.int16)
    
        return dataI,dataQ
        
    def enable_buf(self):
        self.buf_start_reg = 1
        
    def disable_buf(self):
        self.buf_start_reg = 0         
        
class AxisReadoutV1(SocIp):
    # Registers.
    # OUTSEL_REG
    # * 0 : Product of Input Data and DDS.
    # * 1 : DDS.
    # * 2 : Input Data.
    REGISTERS = {'outsel_reg':0}
    
    # Generics.
    NDDS = 8
    
    def __init__(self, ip, **kwargs):
        # Init IP.
        super().__init__(ip)
        
        # Default registers.
        self.outsel_reg=0
        
    def set_out(self,sel="product"):
        if sel is "product":
            self.outsel_reg = 0
        elif sel is "dds":
            self.outsel_reg = 1
        elif sel is "input":
            self.outsel_reg = 2
        else:
            print("AxisReadoutV1: %s output unknown" % sel)
        
class PfbSoc(Overlay):
    FREF_PLL = 204.8
    fs_dac = 384*16
    fs_adc = 384*8
    # Constructor.
    def __init__(self, bitfile, force_init_clks=False,ignore_version=True, **kwargs):
        # Load bitstream.
        super().__init__(bitfile, ignore_version=ignore_version, **kwargs)
        
        # Configure PLLs if requested.
        if force_init_clks:
            self.set_all_clks()
        else:
            rf=self.usp_rf_data_converter_0
            dac_tile = rf.dac_tiles[1] # DAC 228: 0, DAC 229: 1
            DAC_PLL=dac_tile.PLLLockStatus
            adc_tile = rf.adc_tiles[0] # ADC 224: 0, ADC 225: 1, ADC 226: 2, ADC 227: 3
            ADC_PLL=adc_tile.PLLLockStatus
            
            if not (DAC_PLL==2 and ADC_PLL==2):
                self.set_all_clks()
        
        # Create IP objects.
        self.gen0 = AxisSignalGenV3(self.axis_signal_gen_v3_0, 
                                    self.axi_dma_0, 
                                    self.axis_dds_mr_switch_0,
                                    self.axis_switch_0, 
                                    0, 
                                    'HF 0')
        self.gen1 = AxisSignalGenV3(self.axis_signal_gen_v3_1, 
                                    self.axi_dma_0, 
                                    self.axis_dds_mr_switch_1,
                                    self.axis_switch_0, 
                                    1, 
                                    'HF 1')
        self.gen2 = AxisSignalGenV3(self.axis_signal_gen_v3_2, 
                                    self.axi_dma_0, 
                                    self.axis_dds_mr_switch_2,
                                    self.axis_switch_0, 
                                    2, 
                                    'LF 0')
        self.gen3 = AxisSignalGenV3(self.axis_signal_gen_v3_3, 
                                    self.axi_dma_0, 
                                    self.axis_dds_mr_switch_3,
                                    self.axis_switch_0, 
                                    3, 
                                    'LF 1')
        self.tproc  = AxisTProc64x8(self.axis_tproc64_x8_0, self.axi_bram_ctrl_0, self.axi_dma_1)
        self.readout = AxisReadoutV1(self.axis_readout_v1_0)
        
        # This IP drives the full speed buffer
        self.buffer_fs = MrBufferEt(self.mr_buffer_et_0, self.axi_dma_mr_buffer)
        
        # This IP drives the decimated and the accumulated buffers
        self.avg_buf = AxisAvgBuffer(self.axis_avg_buffer_0,
                                     self.axi_dma_avg_buffer_0,
                                     self.axi_dma_avg_buffer_1)
        
    def set_all_clks(self):
        xrfclk.set_all_ref_clks(self.__class__.FREF_PLL)
        
    def setSelection(self,sel):
        self.readout.set_out(sel)
        
    def getFullSpeed(self):
        buff = allocate(shape=(self.buffer_fs.MAX_LENGTH), dtype=np.int32)
        self.buffer_fs.transfer(buff)
        np_buff = np.zeros(self.buffer_fs.MAX_LENGTH, dtype=np.int32)
        np.copyto(np_buff,buff)
        di,dq = format_buffer(np_buff)
        return di, dq

    def getDecimated(self, address=0, length=AxisAvgBuffer.BUF_MAX_LENGTH):
        buff = allocate(shape=length, dtype=np.int32)
        di,dq = self.avg_buf.transfer_buf(buff,address,length)
        return di, dq

    def getAccumulated(self, address=0, length=AxisAvgBuffer.AVG_MAX_LENGTH):
        buff = allocate(shape=length, dtype=np.int64)
        di,dq = self.avg_buf.transfer_avg(buff,address=address,length=length)
        return di, dq
    
    def set_nyquist(self, tile, channel, nqz):
        rf=self.usp_rf_data_converter_0
        
        dac_tile = rf.dac_tiles[tile] # DAC 228: 0, DAC 229: 1
        dac_block = dac_tile.blocks[channel] # CH0: 0, CH1: 1, CH2: 2, CH3: 3
        DAC_Nyq_Zone=dac_block.NyquistZone
        #print("DACNyq_Zone =",DAC_Nyq_Zone)

        dac_block.NyquistZone=nqz
        #DAC_Nyq_Zone_updated=dac_block.NyquistZone
        #print("DACNyq_Zone_Updated =",DAC_Nyq_Zone_updated)
        return dac_block.NyquistZone

