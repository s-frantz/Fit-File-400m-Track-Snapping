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

#import "FITSpeedZoneMesg.h"

@implementation FITSpeedZoneMesg

- (instancetype)init {
    self = [super initWithFitMesgIndex:fit::Profile::MESG_SPEED_ZONE];

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

// HighValue 
- (BOOL)isHighValueValid {
	const fit::Field* field = [super getField:0];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (FITFloat32)getHighValue {
    return ([super getFieldFLOAT32ValueForField:0 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setHighValue:(FITFloat32)highValue {
    [super setFieldFLOAT32ValueForField:0 andValue:(highValue) forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD];
} 

// Name 
- (BOOL)isNameValid {
	const fit::Field* field = [super getField:1];
	if( FIT_NULL == field ) {
		return FALSE;
	}

	return field->IsValueValid() == FIT_TRUE ? TRUE : FALSE;
}

- (NSString *)getName {
    return ([super getFieldSTRINGValueForField:1 forIndex:0 andSubFieldIndex:FIT_SUBFIELD_INDEX_MAIN_FIELD]);
}

- (void)setName:(NSString *)name {
    [super setFieldSTRINGValueForField:1 andValue:(name) forIndex:0];
} 

@end
