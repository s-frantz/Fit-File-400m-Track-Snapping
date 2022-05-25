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


#import "FITMessage+Internal.h"

#import "FITString.h"

#import "FITUserProfileMesg.h"

@implementation FITUserProfileMesg

- (instancetype)init {
    self = [super initWithFitMesgIndex:fit::Profile::MESG_USER_PROFILE];

    return self;
}

// MessageIndex 
- (BOOL)isMessageIndexValid {
	const fit::Field* field = [super getField:254];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITMessageIndex)getMessageIndex {
    return ([super getFieldUINT16ValueForField:254 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setMessageIndex:(FITMessageIndex)messageIndex {
    [super setFieldUINT16ValueForField:254 andValue:(messageIndex) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// FriendlyName 
- (BOOL)isFriendlyNameValid {
	const fit::Field* field = [super getField:0];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (NSString *)getFriendlyName {
    return ([super getFieldSTRINGValueForField:0 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setFriendlyName:(NSString *)friendlyName {
    [super setFieldSTRINGValueForField:0 andValue:(friendlyName) forIndex:0];
} 

// Gender 
- (BOOL)isGenderValid {
	const fit::Field* field = [super getField:1];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITGender)getGender {
    return ([super getFieldENUMValueForField:1 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setGender:(FITGender)gender {
    [super setFieldENUMValueForField:1 andValue:(gender) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Age 
- (BOOL)isAgeValid {
	const fit::Field* field = [super getField:2];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getAge {
    return ([super getFieldUINT8ValueForField:2 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setAge:(FITUInt8)age {
    [super setFieldUINT8ValueForField:2 andValue:(age) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Height 
- (BOOL)isHeightValid {
	const fit::Field* field = [super getField:3];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getHeight {
    return ([super getFieldFLOAT32ValueForField:3 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setHeight:(FITFloat32)height {
    [super setFieldFLOAT32ValueForField:3 andValue:(height) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Weight 
- (BOOL)isWeightValid {
	const fit::Field* field = [super getField:4];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getWeight {
    return ([super getFieldFLOAT32ValueForField:4 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setWeight:(FITFloat32)weight {
    [super setFieldFLOAT32ValueForField:4 andValue:(weight) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Language 
- (BOOL)isLanguageValid {
	const fit::Field* field = [super getField:5];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITLanguage)getLanguage {
    return ([super getFieldENUMValueForField:5 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setLanguage:(FITLanguage)language {
    [super setFieldENUMValueForField:5 andValue:(language) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// ElevSetting 
- (BOOL)isElevSettingValid {
	const fit::Field* field = [super getField:6];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getElevSetting {
    return ([super getFieldENUMValueForField:6 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setElevSetting:(FITDisplayMeasure)elevSetting {
    [super setFieldENUMValueForField:6 andValue:(elevSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// WeightSetting 
- (BOOL)isWeightSettingValid {
	const fit::Field* field = [super getField:7];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getWeightSetting {
    return ([super getFieldENUMValueForField:7 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setWeightSetting:(FITDisplayMeasure)weightSetting {
    [super setFieldENUMValueForField:7 andValue:(weightSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// RestingHeartRate 
- (BOOL)isRestingHeartRateValid {
	const fit::Field* field = [super getField:8];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getRestingHeartRate {
    return ([super getFieldUINT8ValueForField:8 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setRestingHeartRate:(FITUInt8)restingHeartRate {
    [super setFieldUINT8ValueForField:8 andValue:(restingHeartRate) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DefaultMaxRunningHeartRate 
- (BOOL)isDefaultMaxRunningHeartRateValid {
	const fit::Field* field = [super getField:9];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getDefaultMaxRunningHeartRate {
    return ([super getFieldUINT8ValueForField:9 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDefaultMaxRunningHeartRate:(FITUInt8)defaultMaxRunningHeartRate {
    [super setFieldUINT8ValueForField:9 andValue:(defaultMaxRunningHeartRate) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DefaultMaxBikingHeartRate 
- (BOOL)isDefaultMaxBikingHeartRateValid {
	const fit::Field* field = [super getField:10];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getDefaultMaxBikingHeartRate {
    return ([super getFieldUINT8ValueForField:10 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDefaultMaxBikingHeartRate:(FITUInt8)defaultMaxBikingHeartRate {
    [super setFieldUINT8ValueForField:10 andValue:(defaultMaxBikingHeartRate) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DefaultMaxHeartRate 
- (BOOL)isDefaultMaxHeartRateValid {
	const fit::Field* field = [super getField:11];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getDefaultMaxHeartRate {
    return ([super getFieldUINT8ValueForField:11 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDefaultMaxHeartRate:(FITUInt8)defaultMaxHeartRate {
    [super setFieldUINT8ValueForField:11 andValue:(defaultMaxHeartRate) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// HrSetting 
- (BOOL)isHrSettingValid {
	const fit::Field* field = [super getField:12];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayHeart)getHrSetting {
    return ([super getFieldENUMValueForField:12 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setHrSetting:(FITDisplayHeart)hrSetting {
    [super setFieldENUMValueForField:12 andValue:(hrSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// SpeedSetting 
- (BOOL)isSpeedSettingValid {
	const fit::Field* field = [super getField:13];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getSpeedSetting {
    return ([super getFieldENUMValueForField:13 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setSpeedSetting:(FITDisplayMeasure)speedSetting {
    [super setFieldENUMValueForField:13 andValue:(speedSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DistSetting 
- (BOOL)isDistSettingValid {
	const fit::Field* field = [super getField:14];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getDistSetting {
    return ([super getFieldENUMValueForField:14 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDistSetting:(FITDisplayMeasure)distSetting {
    [super setFieldENUMValueForField:14 andValue:(distSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// PowerSetting 
- (BOOL)isPowerSettingValid {
	const fit::Field* field = [super getField:16];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayPower)getPowerSetting {
    return ([super getFieldENUMValueForField:16 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setPowerSetting:(FITDisplayPower)powerSetting {
    [super setFieldENUMValueForField:16 andValue:(powerSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// ActivityClass 
- (BOOL)isActivityClassValid {
	const fit::Field* field = [super getField:17];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITActivityClass)getActivityClass {
    return ([super getFieldENUMValueForField:17 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setActivityClass:(FITActivityClass)activityClass {
    [super setFieldENUMValueForField:17 andValue:(activityClass) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// PositionSetting 
- (BOOL)isPositionSettingValid {
	const fit::Field* field = [super getField:18];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayPosition)getPositionSetting {
    return ([super getFieldENUMValueForField:18 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setPositionSetting:(FITDisplayPosition)positionSetting {
    [super setFieldENUMValueForField:18 andValue:(positionSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// TemperatureSetting 
- (BOOL)isTemperatureSettingValid {
	const fit::Field* field = [super getField:21];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getTemperatureSetting {
    return ([super getFieldENUMValueForField:21 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setTemperatureSetting:(FITDisplayMeasure)temperatureSetting {
    [super setFieldENUMValueForField:21 andValue:(temperatureSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// LocalId 
- (BOOL)isLocalIdValid {
	const fit::Field* field = [super getField:22];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUserLocalId)getLocalId {
    return ([super getFieldUINT16ValueForField:22 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setLocalId:(FITUserLocalId)localId {
    [super setFieldUINT16ValueForField:22 andValue:(localId) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// GlobalId 
- (uint8_t)numGlobalIdValues {
    return [super getFieldNumValuesForField:23 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isGlobalIdValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:23];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITByte)getGlobalIdforIndex:(uint8_t)index {
    return ([super getFieldBYTEValueForField:23 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setGlobalId:(FITByte)globalId forIndex:(uint8_t)index {
    [super setFieldBYTEValueForField:23 andValue:(globalId) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// WakeTime 
- (BOOL)isWakeTimeValid {
	const fit::Field* field = [super getField:28];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITLocaltimeIntoDay)getWakeTime {
    return ([super getFieldUINT32ValueForField:28 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setWakeTime:(FITLocaltimeIntoDay)wakeTime {
    [super setFieldUINT32ValueForField:28 andValue:(wakeTime) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// SleepTime 
- (BOOL)isSleepTimeValid {
	const fit::Field* field = [super getField:29];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITLocaltimeIntoDay)getSleepTime {
    return ([super getFieldUINT32ValueForField:29 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setSleepTime:(FITLocaltimeIntoDay)sleepTime {
    [super setFieldUINT32ValueForField:29 andValue:(sleepTime) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// HeightSetting 
- (BOOL)isHeightSettingValid {
	const fit::Field* field = [super getField:30];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getHeightSetting {
    return ([super getFieldENUMValueForField:30 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setHeightSetting:(FITDisplayMeasure)heightSetting {
    [super setFieldENUMValueForField:30 andValue:(heightSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// UserRunningStepLength 
- (BOOL)isUserRunningStepLengthValid {
	const fit::Field* field = [super getField:31];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getUserRunningStepLength {
    return ([super getFieldFLOAT32ValueForField:31 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setUserRunningStepLength:(FITFloat32)userRunningStepLength {
    [super setFieldFLOAT32ValueForField:31 andValue:(userRunningStepLength) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// UserWalkingStepLength 
- (BOOL)isUserWalkingStepLengthValid {
	const fit::Field* field = [super getField:32];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getUserWalkingStepLength {
    return ([super getFieldFLOAT32ValueForField:32 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setUserWalkingStepLength:(FITFloat32)userWalkingStepLength {
    [super setFieldFLOAT32ValueForField:32 andValue:(userWalkingStepLength) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DepthSetting 
- (BOOL)isDepthSettingValid {
	const fit::Field* field = [super getField:47];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDisplayMeasure)getDepthSetting {
    return ([super getFieldENUMValueForField:47 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDepthSetting:(FITDisplayMeasure)depthSetting {
    [super setFieldENUMValueForField:47 andValue:(depthSetting) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// DiveCount 
- (BOOL)isDiveCountValid {
	const fit::Field* field = [super getField:49];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt32)getDiveCount {
    return ([super getFieldUINT32ValueForField:49 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setDiveCount:(FITUInt32)diveCount {
    [super setFieldUINT32ValueForField:49 andValue:(diveCount) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

@end
