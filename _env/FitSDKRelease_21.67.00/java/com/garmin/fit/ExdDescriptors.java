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


public enum ExdDescriptors  {
    BIKE_LIGHT_BATTERY_STATUS((short)0),
    BEAM_ANGLE_STATUS((short)1),
    BATERY_LEVEL((short)2),
    LIGHT_NETWORK_MODE((short)3),
    NUMBER_LIGHTS_CONNECTED((short)4),
    CADENCE((short)5),
    DISTANCE((short)6),
    ESTIMATED_TIME_OF_ARRIVAL((short)7),
    HEADING((short)8),
    TIME((short)9),
    BATTERY_LEVEL((short)10),
    TRAINER_RESISTANCE((short)11),
    TRAINER_TARGET_POWER((short)12),
    TIME_SEATED((short)13),
    TIME_STANDING((short)14),
    ELEVATION((short)15),
    GRADE((short)16),
    ASCENT((short)17),
    DESCENT((short)18),
    VERTICAL_SPEED((short)19),
    DI2_BATTERY_LEVEL((short)20),
    FRONT_GEAR((short)21),
    REAR_GEAR((short)22),
    GEAR_RATIO((short)23),
    HEART_RATE((short)24),
    HEART_RATE_ZONE((short)25),
    TIME_IN_HEART_RATE_ZONE((short)26),
    HEART_RATE_RESERVE((short)27),
    CALORIES((short)28),
    GPS_ACCURACY((short)29),
    GPS_SIGNAL_STRENGTH((short)30),
    TEMPERATURE((short)31),
    TIME_OF_DAY((short)32),
    BALANCE((short)33),
    PEDAL_SMOOTHNESS((short)34),
    POWER((short)35),
    FUNCTIONAL_THRESHOLD_POWER((short)36),
    INTENSITY_FACTOR((short)37),
    WORK((short)38),
    POWER_RATIO((short)39),
    NORMALIZED_POWER((short)40),
    TRAINING_STRESS_SCORE((short)41),
    TIME_ON_ZONE((short)42),
    SPEED((short)43),
    LAPS((short)44),
    REPS((short)45),
    WORKOUT_STEP((short)46),
    COURSE_DISTANCE((short)47),
    NAVIGATION_DISTANCE((short)48),
    COURSE_ESTIMATED_TIME_OF_ARRIVAL((short)49),
    NAVIGATION_ESTIMATED_TIME_OF_ARRIVAL((short)50),
    COURSE_TIME((short)51),
    NAVIGATION_TIME((short)52),
    COURSE_HEADING((short)53),
    NAVIGATION_HEADING((short)54),
    POWER_ZONE((short)55),
    TORQUE_EFFECTIVENESS((short)56),
    TIMER_TIME((short)57),
    POWER_WEIGHT_RATIO((short)58),
    LEFT_PLATFORM_CENTER_OFFSET((short)59),
    RIGHT_PLATFORM_CENTER_OFFSET((short)60),
    LEFT_POWER_PHASE_START_ANGLE((short)61),
    RIGHT_POWER_PHASE_START_ANGLE((short)62),
    LEFT_POWER_PHASE_FINISH_ANGLE((short)63),
    RIGHT_POWER_PHASE_FINISH_ANGLE((short)64),
    GEARS((short)65),
    PACE((short)66),
    TRAINING_EFFECT((short)67),
    VERTICAL_OSCILLATION((short)68),
    VERTICAL_RATIO((short)69),
    GROUND_CONTACT_TIME((short)70),
    LEFT_GROUND_CONTACT_TIME_BALANCE((short)71),
    RIGHT_GROUND_CONTACT_TIME_BALANCE((short)72),
    STRIDE_LENGTH((short)73),
    RUNNING_CADENCE((short)74),
    PERFORMANCE_CONDITION((short)75),
    COURSE_TYPE((short)76),
    TIME_IN_POWER_ZONE((short)77),
    NAVIGATION_TURN((short)78),
    COURSE_LOCATION((short)79),
    NAVIGATION_LOCATION((short)80),
    COMPASS((short)81),
    GEAR_COMBO((short)82),
    MUSCLE_OXYGEN((short)83),
    ICON((short)84),
    COMPASS_HEADING((short)85),
    GPS_HEADING((short)86),
    GPS_ELEVATION((short)87),
    ANAEROBIC_TRAINING_EFFECT((short)88),
    COURSE((short)89),
    OFF_COURSE((short)90),
    GLIDE_RATIO((short)91),
    VERTICAL_DISTANCE((short)92),
    VMG((short)93),
    AMBIENT_PRESSURE((short)94),
    PRESSURE((short)95),
    VAM((short)96),
    INVALID((short)255);

    protected short value;

    private ExdDescriptors(short value) {
        this.value = value;
    }

    public static ExdDescriptors getByValue(final Short value) {
        for (final ExdDescriptors type : ExdDescriptors.values()) {
            if (value == type.value)
                return type;
        }

        return ExdDescriptors.INVALID;
    }

    /**
     * Retrieves the String Representation of the Value
     * @return The string representation of the value
     */
    public static String getStringFromValue( ExdDescriptors value ) {
        return value.name();
    }

    public short getValue() {
        return value;
    }


}
