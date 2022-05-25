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


#import <Foundation/Foundation.h>

#import "FITMessage.h"
#import "FITTypes.h"

NS_ASSUME_NONNULL_BEGIN

@interface FITExdScreenConfigurationMesg : FITMessage
- (id)init;
// ScreenIndex 
- (BOOL)isScreenIndexValid;
- (FITUInt8)getScreenIndex;
- (void)setScreenIndex:(FITUInt8)screenIndex;
// FieldCount 
- (BOOL)isFieldCountValid;
- (FITUInt8)getFieldCount;
- (void)setFieldCount:(FITUInt8)fieldCount;
// Layout 
- (BOOL)isLayoutValid;
- (FITExdLayout)getLayout;
- (void)setLayout:(FITExdLayout)layout;
// ScreenEnabled 
- (BOOL)isScreenEnabledValid;
- (FITBool)getScreenEnabled;
- (void)setScreenEnabled:(FITBool)screenEnabled;

@end

NS_ASSUME_NONNULL_END
