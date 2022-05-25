////////////////////////////////////////////////////////////////////////////////
// The following FIT Protocol software provided may be used with FIT protocol
// devices only and remains the copyrighted property of Garmin Canada Inc.
// The software is being provided on an "as-is" basis and as an accommodation,
// and therefore all warranties, representations, or guarantees of any kind
// (whether express, implied or statutory) including, without limitation,
// warranties of merchantability, non-infringement, or fitness for a particular
// purpose, are specifically disclaimed.
//
// Copyright 2021 Garmin International, Inc.
////////////////////////////////////////////////////////////////////////////////
// ****WARNING****  This file is auto-generated!  Do NOT edit this file.
// Profile Version = 21.67Release
// Tag = production/akw/21.67.00-0-gd790f76b
////////////////////////////////////////////////////////////////////////////////


package com.garmin.fit;


public enum DiveAlarmType  {
    DEPTH((short)0),
    TIME((short)1),
    INVALID((short)255);

    protected short value;

    private DiveAlarmType(short value) {
        this.value = value;
    }

    public static DiveAlarmType getByValue(final Short value) {
        for (final DiveAlarmType type : DiveAlarmType.values()) {
            if (value == type.value)
                return type;
        }

        return DiveAlarmType.INVALID;
    }

    /**
     * Retrieves the String Representation of the Value
     * @return The string representation of the value
     */
    public static String getStringFromValue( DiveAlarmType value ) {
        return value.name();
    }

    public short getValue() {
        return value;
    }


}
