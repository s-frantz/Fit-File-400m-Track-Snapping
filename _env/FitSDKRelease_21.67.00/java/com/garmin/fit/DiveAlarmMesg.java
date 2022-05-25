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



public class DiveAlarmMesg extends Mesg   {

    
    public static final int MessageIndexFieldNum = 254;
    
    public static final int DepthFieldNum = 0;
    
    public static final int TimeFieldNum = 1;
    
    public static final int EnabledFieldNum = 2;
    
    public static final int AlarmTypeFieldNum = 3;
    
    public static final int SoundFieldNum = 4;
    
    public static final int DiveTypesFieldNum = 5;
    

    protected static final  Mesg diveAlarmMesg;
    static {
        // dive_alarm
        diveAlarmMesg = new Mesg("dive_alarm", MesgNum.DIVE_ALARM);
        diveAlarmMesg.addField(new Field("message_index", MessageIndexFieldNum, 132, 1, 0, "", false, Profile.Type.MESSAGE_INDEX));
        
        diveAlarmMesg.addField(new Field("depth", DepthFieldNum, 134, 1000, 0, "m", false, Profile.Type.UINT32));
        
        diveAlarmMesg.addField(new Field("time", TimeFieldNum, 133, 1, 0, "s", false, Profile.Type.SINT32));
        
        diveAlarmMesg.addField(new Field("enabled", EnabledFieldNum, 0, 1, 0, "", false, Profile.Type.BOOL));
        
        diveAlarmMesg.addField(new Field("alarm_type", AlarmTypeFieldNum, 0, 1, 0, "", false, Profile.Type.DIVE_ALARM_TYPE));
        
        diveAlarmMesg.addField(new Field("sound", SoundFieldNum, 0, 1, 0, "", false, Profile.Type.TONE));
        
        diveAlarmMesg.addField(new Field("dive_types", DiveTypesFieldNum, 0, 1, 0, "", false, Profile.Type.SUB_SPORT));
        
    }

    public DiveAlarmMesg() {
        super(Factory.createMesg(MesgNum.DIVE_ALARM));
    }

    public DiveAlarmMesg(final Mesg mesg) {
        super(mesg);
    }


    /**
     * Get message_index field
     * Comment: Index of the alarm
     *
     * @return message_index
     */
    public Integer getMessageIndex() {
        return getFieldIntegerValue(254, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Set message_index field
     * Comment: Index of the alarm
     *
     * @param messageIndex
     */
    public void setMessageIndex(Integer messageIndex) {
        setFieldValue(254, 0, messageIndex, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get depth field
     * Units: m
     *
     * @return depth
     */
    public Float getDepth() {
        return getFieldFloatValue(0, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Set depth field
     * Units: m
     *
     * @param depth
     */
    public void setDepth(Float depth) {
        setFieldValue(0, 0, depth, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get time field
     * Units: s
     *
     * @return time
     */
    public Integer getTime() {
        return getFieldIntegerValue(1, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Set time field
     * Units: s
     *
     * @param time
     */
    public void setTime(Integer time) {
        setFieldValue(1, 0, time, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get enabled field
     *
     * @return enabled
     */
    public Bool getEnabled() {
        Short value = getFieldShortValue(2, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
        if (value == null) {
            return null;
        }
        return Bool.getByValue(value);
    }

    /**
     * Set enabled field
     *
     * @param enabled
     */
    public void setEnabled(Bool enabled) {
        setFieldValue(2, 0, enabled.value, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get alarm_type field
     *
     * @return alarm_type
     */
    public DiveAlarmType getAlarmType() {
        Short value = getFieldShortValue(3, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
        if (value == null) {
            return null;
        }
        return DiveAlarmType.getByValue(value);
    }

    /**
     * Set alarm_type field
     *
     * @param alarmType
     */
    public void setAlarmType(DiveAlarmType alarmType) {
        setFieldValue(3, 0, alarmType.value, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get sound field
     *
     * @return sound
     */
    public Tone getSound() {
        Short value = getFieldShortValue(4, 0, Fit.SUBFIELD_INDEX_MAIN_FIELD);
        if (value == null) {
            return null;
        }
        return Tone.getByValue(value);
    }

    /**
     * Set sound field
     *
     * @param sound
     */
    public void setSound(Tone sound) {
        setFieldValue(4, 0, sound.value, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    public SubSport[] getDiveTypes() {
        
        Short[] values = getFieldShortValues(5, Fit.SUBFIELD_INDEX_MAIN_FIELD);
        SubSport[] rv = new SubSport[values.length];
        for(int i = 0; i < values.length; i++){
            rv[i] = SubSport.getByValue(values[i]);
        }
        return rv;
        
    }

    /**
     * @return number of dive_types
     */
    public int getNumDiveTypes() {
        return getNumFieldValues(5, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

    /**
     * Get dive_types field
     *
     * @param index of dive_types
     * @return dive_types
     */
    public SubSport getDiveTypes(int index) {
        Short value = getFieldShortValue(5, index, Fit.SUBFIELD_INDEX_MAIN_FIELD);
        if (value == null) {
            return null;
        }
        return SubSport.getByValue(value);
    }

    /**
     * Set dive_types field
     *
     * @param index of dive_types
     * @param diveTypes
     */
    public void setDiveTypes(int index, SubSport diveTypes) {
        setFieldValue(5, index, diveTypes.value, Fit.SUBFIELD_INDEX_MAIN_FIELD);
    }

}
