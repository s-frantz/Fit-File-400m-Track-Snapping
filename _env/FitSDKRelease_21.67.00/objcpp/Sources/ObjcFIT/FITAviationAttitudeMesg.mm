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


#import "FITAviationAttitudeMesg.h"

@implementation FITAviationAttitudeMesg

- (instancetype)init {
    self = [super initWithFitMesgIndex:fit::Profile::MESG_AVIATION_ATTITUDE];

    return self;
}

// Timestamp 
- (BOOL)isTimestampValid {
	const fit::Field* field = [super getField:253];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITDate *)getTimestamp {
    return FITDateFromTimestamp([super getFieldUINT32ValueForField:253 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setTimestamp:(FITDate *)timestamp {
    [super setFieldUINT32ValueForField:253 andValue:TimestampFromFITDate(timestamp) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// TimestampMs 
- (BOOL)isTimestampMsValid {
	const fit::Field* field = [super getField:0];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt16)getTimestampMs {
    return ([super getFieldUINT16ValueForField:0 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setTimestampMs:(FITUInt16)timestampMs {
    [super setFieldUINT16ValueForField:0 andValue:(timestampMs) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// SystemTime 
- (uint8_t)numSystemTimeValues {
    return [super getFieldNumValuesForField:1 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isSystemTimeValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:1];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt32)getSystemTimeforIndex:(uint8_t)index {
    return ([super getFieldUINT32ValueForField:1 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setSystemTime:(FITUInt32)systemTime forIndex:(uint8_t)index {
    [super setFieldUINT32ValueForField:1 andValue:(systemTime) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Pitch 
- (uint8_t)numPitchValues {
    return [super getFieldNumValuesForField:2 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isPitchValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:2];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getPitchforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:2 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setPitch:(FITFloat32)pitch forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:2 andValue:(pitch) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Roll 
- (uint8_t)numRollValues {
    return [super getFieldNumValuesForField:3 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isRollValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:3];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getRollforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:3 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setRoll:(FITFloat32)roll forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:3 andValue:(roll) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// AccelLateral 
- (uint8_t)numAccelLateralValues {
    return [super getFieldNumValuesForField:4 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isAccelLateralValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:4];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getAccelLateralforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:4 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setAccelLateral:(FITFloat32)accelLateral forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:4 andValue:(accelLateral) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// AccelNormal 
- (uint8_t)numAccelNormalValues {
    return [super getFieldNumValuesForField:5 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isAccelNormalValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:5];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getAccelNormalforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:5 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setAccelNormal:(FITFloat32)accelNormal forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:5 andValue:(accelNormal) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// TurnRate 
- (uint8_t)numTurnRateValues {
    return [super getFieldNumValuesForField:6 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isTurnRateValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:6];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getTurnRateforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:6 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setTurnRate:(FITFloat32)turnRate forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:6 andValue:(turnRate) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Stage 
- (uint8_t)numStageValues {
    return [super getFieldNumValuesForField:7 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isStageValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:7];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITAttitudeStage)getStageforIndex:(uint8_t)index {
    return ([super getFieldENUMValueForField:7 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setStage:(FITAttitudeStage)stage forIndex:(uint8_t)index {
    [super setFieldENUMValueForField:7 andValue:(stage) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// AttitudeStageComplete 
- (uint8_t)numAttitudeStageCompleteValues {
    return [super getFieldNumValuesForField:8 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isAttitudeStageCompleteValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:8];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITUInt8)getAttitudeStageCompleteforIndex:(uint8_t)index {
    return ([super getFieldUINT8ValueForField:8 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setAttitudeStageComplete:(FITUInt8)attitudeStageComplete forIndex:(uint8_t)index {
    [super setFieldUINT8ValueForField:8 andValue:(attitudeStageComplete) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Track 
- (uint8_t)numTrackValues {
    return [super getFieldNumValuesForField:9 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isTrackValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:9];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getTrackforIndex:(uint8_t)index {
    return ([super getFieldFLOAT32ValueForField:9 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setTrack:(FITFloat32)track forIndex:(uint8_t)index {
    [super setFieldFLOAT32ValueForField:9 andValue:(track) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Validity 
- (uint8_t)numValidityValues {
    return [super getFieldNumValuesForField:10 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
}

- (BOOL)isValidityValidforIndex:(uint8_t)index {
	const fit::Field* field = [super getField:10];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid(index) == FIT_TRUE ? TRUE : FALSE;
}

- (FITAttitudeValidity)getValidityforIndex:(uint8_t)index {
    return ([super getFieldUINT16ValueForField:10 forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setValidity:(FITAttitudeValidity)validity forIndex:(uint8_t)index {
    [super setFieldUINT16ValueForField:10 andValue:(validity) forIndex:index andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

@end
