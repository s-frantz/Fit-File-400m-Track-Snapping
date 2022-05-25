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

import java.util.HashMap;
import java.util.Map;

public class FlyeExerciseName  {
    public static final int CABLE_CROSSOVER = 0;
    public static final int DECLINE_DUMBBELL_FLYE = 1;
    public static final int DUMBBELL_FLYE = 2;
    public static final int INCLINE_DUMBBELL_FLYE = 3;
    public static final int KETTLEBELL_FLYE = 4;
    public static final int KNEELING_REAR_FLYE = 5;
    public static final int SINGLE_ARM_STANDING_CABLE_REVERSE_FLYE = 6;
    public static final int SWISS_BALL_DUMBBELL_FLYE = 7;
    public static final int ARM_ROTATIONS = 8;
    public static final int HUG_A_TREE = 9;
    public static final int INVALID = Fit.UINT16_INVALID;

    private static final Map<Integer, String> stringMap;

    static {
        stringMap = new HashMap<Integer, String>();
        stringMap.put(CABLE_CROSSOVER, "CABLE_CROSSOVER");
        stringMap.put(DECLINE_DUMBBELL_FLYE, "DECLINE_DUMBBELL_FLYE");
        stringMap.put(DUMBBELL_FLYE, "DUMBBELL_FLYE");
        stringMap.put(INCLINE_DUMBBELL_FLYE, "INCLINE_DUMBBELL_FLYE");
        stringMap.put(KETTLEBELL_FLYE, "KETTLEBELL_FLYE");
        stringMap.put(KNEELING_REAR_FLYE, "KNEELING_REAR_FLYE");
        stringMap.put(SINGLE_ARM_STANDING_CABLE_REVERSE_FLYE, "SINGLE_ARM_STANDING_CABLE_REVERSE_FLYE");
        stringMap.put(SWISS_BALL_DUMBBELL_FLYE, "SWISS_BALL_DUMBBELL_FLYE");
        stringMap.put(ARM_ROTATIONS, "ARM_ROTATIONS");
        stringMap.put(HUG_A_TREE, "HUG_A_TREE");
    }


    /**
     * Retrieves the String Representation of the Value
     * @return The string representation of the value, or empty if unknown
     */
    public static String getStringFromValue( Integer value ) {
        if( stringMap.containsKey( value ) ) {
            return stringMap.get( value );
        }

        return "";
    }

    /**
     * Retrieves a value given a string representation
     * @return The value or INVALID if unkwown
     */
    public static Integer getValueFromString( String value ) {
        for( Map.Entry<Integer, String> entry : stringMap.entrySet() ) {
            if( entry.getValue().equals( value ) ) {
                return entry.getKey();
            }
        }

        return INVALID;
    }

}
