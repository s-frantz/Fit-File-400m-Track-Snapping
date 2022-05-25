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
#import "FitDeveloperField.h"

@interface FitDeveloperField ()

@end

@implementation FitDeveloperField

- (FIT_UINT8) Write:(FILE*) file forDeveloperFieldDef:(const fit::DeveloperFieldDefinition *)fieldDef
{
    FIT_UINT8 byte;

    byte = fieldDef->GetNum();
    fwrite(&byte, 1, 1, file);
    byte = fieldDef->GetSize();
    fwrite(&byte, 1, 1, file);
    byte = fieldDef->GetDeveloperDataIndex();
    fwrite(&byte, 1, 1, file);

    return 3;
}

@end
