#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Copyright (c) 2018,2019,2020,2021 Stephen Goadhouse <sgoadhouse@virginia.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#---------------------------------------------------------------------------------
#  Control of HP/Agilent/Keysight MSO-X/DSO-X 3000A Oscilloscope using
#  standard SCPI commands with PyVISA
#
# For more information on SCPI, see:
# https://en.wikipedia.org/wiki/Standard_Commands_for_Programmable_Instruments
# http://www.ivifoundation.org/docs/scpi-99.pdf
#-------------------------------------------------------------------------------

# For future Python3 compatibility:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import sleep
from sys import version_info
import pyvisa as visa

class SCPI(object):
    """Basic class for controlling and accessing an Oscilloscope with Standard SCPI Commands"""

    OverRange = +9.9E+37                  # Number which indicates Over Range
    UnderRange = -9.9E+37                 # Number which indicates Under Range

    def __init__(self, resource, max_chan=1, wait=0,
                     cmd_prefix = '',
                     read_strip = '',
                     read_termination = '',
                     write_termination = '',
                     timeout = 15000):
        """Init the class with the instruments resource string

        resource   - resource string or VISA descriptor, like TCPIP0::172.16.2.13::INSTR
        max_chan   - number of channels
        wait       - float that gives the default number of seconds to wait after sending each command
        cmd_prefix - optional command prefix (ie. some instruments require a ':' prefix)
        read_strip        - optional read_strip parameter used to strip any returned termination characters
        read_termination  - optional read_termination parameter to pass to open_resource()
        write_termination - optional write_termination parameter to pass to open_resource()
        """
        self._resource = resource
        self._max_chan = max_chan                # number of channels
        self._wait = wait
        self._prefix = cmd_prefix
        self._curr_chan = 1                      # set the current channel to the first one
        self._read_strip = read_strip
        self._read_termination = read_termination
        self._write_termination = write_termination
        self._timeout = timeout
        self._inst = None

    def open(self):
        """Open a connection to the VISA device with PYVISA-py python library"""
        self._rm = visa.ResourceManager('@py')
        self._inst = self._rm.open_resource(self._resource,
                                            read_termination=self._read_termination,
                                            write_termination=self._write_termination)
        self._inst.timeout = self._timeout

        # Keysight recommends using clear() but it is currently not implemented in pyvisa-py v0.2
        # self._inst.clear()
        #
        # Instead, send a *CLS system command to at least clear the command
        # handler (error queues and such)
        self.clear()

    def close(self):
        """Close the VISA connection"""
        self._inst.close()

    @property
    def channel(self):
        return self._curr_chan

    @channel.setter
    def channel(self, value):
        self._curr_chan = value

    def _instQuery(self, queryStr, checkErrors=True):
        if (queryStr[0] != '*'):
            queryStr = self._prefix + queryStr
        #print("QUERY:",queryStr)
        result = self._inst.query(queryStr)
        if checkErrors:
            self.checkInstErrors(queryStr)
        return result.rstrip(self._read_strip)

    def _instQueryNumber(self, queryStr, checkErrors=True):
        return float(self._instQuery(queryStr, checkErrors))

    def _instWrite(self, writeStr, checkErrors=True):
        if (writeStr[0] != '*'):
            writeStr = self._prefix + writeStr
        #print("WRITE:",writeStr)
        result = self._inst.write(writeStr)
        if checkErrors:
            self.checkInstErrors(writeStr)
        return result

    def _chStr(self, channel):
        """return the channel string given the channel number and using the format CHx"""

        return 'CH{}'.format(channel)

    def _chanStr(self, channel):
        """return the channel string given the channel number and using the format x"""

        return '{}'.format(channel)

    def _channelStr(self, channel):
        """return the channel string given the channel number and using the format CHANnelx if x is numeric"""

        try:
            return 'CHANnel{}'.format(int(channel))
        except ValueError:
            return self._chanStr(channel)

    def _onORoff(self, str):
        """Check if string says it is ON or OFF and return True if ON
        and False if OFF
        """

        # Only check first two characters so do not need to deal with
        # trailing whitespace and such
        if str[:2] == 'ON':
            return True
        else:
            return False

    def _1OR0(self, str):
        """Check if string says it is 1 or 0 and return True if 1
        and False if 0
        """

        # Only check first character so do not need to deal with
        # trailing whitespace and such
        if str[:1] == '1':
            return True
        else:
            return False

    def _chanNumber(self, str):
        """Decode the response as a channel number and return it. Return 0 if string does not decode properly.
        """

        # Only check first character so do not need to deal with
        # trailing whitespace and such
        if str[:4] == 'CHAN':
            return int(str[4])
        else:
            return 0

    def _wait(self):
        """Wait until all preceeding commands complete"""
        #self._instWrite('*WAI')
        self._instWrite('*OPC')
        wait = True
        while(wait):
            ret = self._instQuery('*OPC?')
            if ret[0] == '1':
                wait = False

    # =========================================================
    # Taken from the MSO-X 3000 Programming Guide and modified to work
    # within this class ...
    # =========================================================
    # Check for instrument errors:
    # =========================================================
    def checkInstErrors(self, commandStr):

        errors = False
        while True:
            # checkErrors=False prevents infinite recursion!
            error_string = self._instQuery("SYSTem:ERRor?", checkErrors=False)
            error_string = error_string.strip()  # remove trailing and leading whitespace
            if error_string: # If there is an error string value.
                if error_string.find("+0,", 0, 3) == -1:
                    # Not "No error".
                    print("ERROR: {}, command: '{}'".format(error_string, commandStr))
                    #print "Exited because of error."
                    #sys.exit(1)
                    errors = True           # indicate there was an error
                else: # "No error"
                    break

            else: # :SYSTem:ERRor? should always return string.
                print("ERROR: :SYSTem:ERRor? returned nothing, command: '{}'".format(commandStr))
                #print "Exited because of error."
                #sys.exit(1)
                errors = True # if unexpected response, then set as Error
                break

        return errors           # indicate if there was an error

    # =========================================================
    # Based on do_query_ieee_block() from the MSO-X 3000 Programming
    # Guide and modified to work within this class ...
    # =========================================================
    def _instQueryIEEEBlock(self, queryStr):
        if (queryStr[0] != '*'):
            queryStr = self._prefix + queryStr
        #print("QUERYIEEEBlock:",queryStr)
        result = self._inst.query_binary_values(queryStr, datatype='s', container=bytes)
        self.checkInstErrors(queryStr)
        return result

    # =========================================================
    # Based on code from the MSO-X 3000 Programming
    # Guide and modified to work within this class ...
    # =========================================================
    def _instQueryNumbers(self, queryStr):
        if (queryStr[0] != '*'):
            queryStr = self._prefix + queryStr
        #print("QUERYNumbers:",queryStr)
        result = self._inst.query_ascii_values(queryStr, converter='f', separator=',')
        self.checkInstErrors(queryStr)
        return result

    # =========================================================
    # Based on do_command_ieee_block() from the MSO-X 3000 Programming
    # Guide and modified to work within this class ...
    # =========================================================
    def _instWriteIEEEBlock(self, writeStr, values):
        if (writeStr[0] != '*'):
            writeStr = self._prefix + writeStr
        #print("WRITE:",writeStr)

        if (version_info < (3,)):
            ## If PYTHON 2, must use datatype of 'c'
            datatype = 'c'
        else:
            ## If PYTHON 2, must use datatype of 'B' to get the same result
            datatype = 'B'

        result = self._inst.write_binary_values(writeStr, values, datatype=datatype)
        self.checkInstErrors(writeStr)
        return result

    def _instWriteIEEENumbers(self, writeStr, values):
        if (writeStr[0] != '*'):
            writeStr = self._prefix + writeStr
        #print("WRITE:",writeStr)

        result = self._inst.write_binary_values(writeStr, values, datatype='f')
        self.checkInstErrors(writeStr)
        return result

    def idn(self):
        """Return response to *IDN? message"""
        return self._instQuery('*IDN?')

    def clear(self):
        """Sends a *CLS message to clear status and error queues"""
        return self._instWrite('*CLS')

    def reset(self):
        """Sends a *RST message to reset to defaults"""
        return self._instWrite('*RST')

    def setLocal(self):
        """Set the power supply to LOCAL mode where front panel keys work again
        """

        # Not sure if this is SCPI, but it appears to be supported
        # across different instruments
        self._instWrite('SYSTem:LOCK OFF')

    def setRemote(self):
        """Set the power supply to REMOTE mode where it is controlled via VISA
        """

        # Not sure if this is SCPI, but it appears to be supported
        # across different instruments
        self._instWrite('SYSTem:LOCK ON')

    def setRemoteLock(self):
        """Set the power supply to REMOTE Lock mode where it is
           controlled via VISA & front panel is locked out
        """

        # Not sure if this is SCPI, but it appears to be supported
        # across different instruments
        self._instWrite('SYSTem:LOCK ON')

    def beeperOn(self):
        """Enable the system beeper for the instrument"""
        # no beeper to turn off, so make it do nothing
        pass

    def beeperOff(self):
        """Disable the system beeper for the instrument"""
        # no beeper to turn off, so make it do nothing
        pass

    def isOutputOn(self, channel=None):
        """Return true if the output of channel is ON, else false

           channel - number of the channel starting at 1
        """

        # If a channel number is passed in, make it the
        # current channel
        if channel is not None:
            self.channel = channel

        str = 'STATus? {}'.format(self._channelStr(self.channel))
        ret = self._instQuery(str)
        # @@@print("1:", ret)
        return self._1OR0(ret)

    def outputOn(self, channel=None, wait=None):
        """Turn on the output for channel

           wait    - number of seconds to wait after sending command
           channel - number of the channel starting at 1
        """

        # If a channel number is passed in, make it the
        # current channel
        if channel is not None:
            self.channel = channel

        # If a wait time is NOT passed in, set wait to the
        # default time
        if wait is None:
            wait = self._wait

        str = 'VIEW {}'.format(self._channelStr(self.channel))
        self._instWrite(str)
        sleep(wait)

    def outputOff(self, channel=None, wait=None):
        """Turn off the output for channel

           channel - number of the channel starting at 1
        """

        # If a channel number is passed in, make it the
        # current channel
        if channel is not None:
            self.channel = channel

        # If a wait time is NOT passed in, set wait to the
        # default time
        if wait is None:
            wait = self._wait

        str = 'BLANK {}'.format(self._channelStr(self.channel))
        self._instWrite(str)
        sleep(wait)

    def outputOnAll(self, wait=None):
        """Turn on the output for ALL channels

        """

        # If a wait time is NOT passed in, set wait to the
        # default time
        if wait is None:
            wait = self._wait

        for chan in range(1,self._max_chan+1):
            str = 'VIEW {}'.format(self._channelStr(chan))
            self._instWrite(str)

        sleep(wait)

    def outputOffAll(self, wait=None):
        """Turn off the output for ALL channels

        """

        # If a wait time is NOT passed in, set wait to the
        # default time
        if wait is None:
            wait = self._wait

        #for chan in range(1,self._max_chan+1):
        #    str = 'BLANK {}'.format(self._channelStr(chan))
        #    self._instWrite(str)

        # Blank without a parameter turns off ALL sources
        str = 'BLANK'
        self._instWrite(str)

        sleep(wait)             # give some time for PS to respond

    def measureVoltage(self, channel=None):
        """Read and return a voltage measurement from channel

           channel - number of the channel starting at 1
        """

        # If a channel number is passed in, make it the
        # current channel
        if channel is not None:
            self.channel = channel

        str = 'INSTrument:NSELect {}; MEASure:VOLTage:DC?'.format(self.channel)
        val = self._instQueryNumber(str)
        return val
