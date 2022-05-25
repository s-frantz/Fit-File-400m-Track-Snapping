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


#if !defined(FIT_DECODE_HPP)
#define FIT_DECODE_HPP

#include <iosfwd>
#include <string>
#include <unordered_map>
#include "fit.hpp"
#include "fit_accumulator.hpp"
#include "fit_field.hpp"
#include "fit_mesg.hpp"
#include "fit_mesg_definition.hpp"
#include "fit_mesg_definition_listener.hpp"
#include "fit_developer_field_description_listener.hpp"
#include "fit_mesg_listener.hpp"
#include "fit_runtime_exception.hpp"
#include "fit_developer_data_id_mesg.hpp"

namespace fit
{
class Decode
{
public:
    Decode();

    FIT_BOOL IsFIT(std::istream &file);
    ///////////////////////////////////////////////////////////////////////
    // Reads the file header to check if the file is FIT.
    // Does not check CRC.
    // Parameters:
    //    file     Pointer to file to read.
    // Returns true if file is FIT.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL CheckIntegrity(std::istream &file);
    ///////////////////////////////////////////////////////////////////////
    // Reads the FIT binary file header and crc to check compatibility and integrity.
    // Parameters:
    //    file     Pointer to file to read.
    // Returns true if file is ok (not corrupt).
    ///////////////////////////////////////////////////////////////////////

    void SkipHeader();
    ///////////////////////////////////////////////////////////////////////
    // Overrides the default read behaviour by skipping header decode.
    // CRC checking is not possible since the datasize is unknown.
    // Decode continues until EOF is encountered or a decode error occurs.
    // May only be called prior to calling Read.
    ///////////////////////////////////////////////////////////////////////

    void IncompleteStream();
    ///////////////////////////////////////////////////////////////////////
    // Override the default read behaviour allowing decode of partial streams.
    // If EOF is encountered no exception is raised. Caller may choose to call
    // resume possibly after more bytes have arrived in the stream. May only be set
    // prior to first calling Read.
    ///////////////////////////////////////////////////////////////////////

    void SuppressComponentExpansion(void);
    ///////////////////////////////////////////////////////////////////////
    // Override the default read behaviour by suppressing the component expansion
    // If your application does not care about component expansion this can speed
    // up processing significantly.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL Read(std::istream &file, MesgListener& mesgListener);
    ///////////////////////////////////////////////////////////////////////
    // Reads a FIT binary file.
    // Parameters:
    //    file                    Pointer to file to read.
    //    mesgListener            Message listener
    // Returns true if finished read file, otherwise false if decoding is paused.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL Read(std::istream &file, MesgListener& mesgListener, MesgDefinitionListener& mesgDefinitionListener);
    ///////////////////////////////////////////////////////////////////////
    // Reads a FIT binary file.
    // Parameters:
    //    file                    Pointer to file to read.
    //    mesgListener            Message listener
    //    mesgDefinitionListener  Message definition listener
    // Returns true if finished read file, otherwise false if decoding is paused.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL Read
        (
        std::istream* file,
        MesgListener* mesgListener,
        MesgDefinitionListener* definitionListener,
        DeveloperFieldDescriptionListener* descriptionListener
        );
    ///////////////////////////////////////////////////////////////////////
    // Reads a FIT binary file.
    // Parameters:
    //    file                    Pointer to file to read.
    //    mesgListener            Message listener
    //    definitionListener      Message definition listener
    //    descriptionListener     Developer field description listener
    // Returns true if finished read file, otherwise false if decoding is paused.
    ///////////////////////////////////////////////////////////////////////


    void Pause(void);
    ///////////////////////////////////////////////////////////////////////
    // Pauses the decoding of a FIT binary file.  Call Resume() to resume decoding.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL Resume(void);
    ///////////////////////////////////////////////////////////////////////
    // Resumes the decoding of a FIT binary file (see Pause()).
    // Returns true if finished reading file.
    ///////////////////////////////////////////////////////////////////////

    FIT_BOOL getInvalidDataSize(void);
    ///////////////////////////////////////////////////////////////////////
    // Returns the invalid data size flag.
    // This flag is set when the file size in the header is 0.
    ///////////////////////////////////////////////////////////////////////

    void setInvalidDataSize(FIT_BOOL value);
    ///////////////////////////////////////////////////////////////////////
    // Set the invalid data size flag.
    // Parameters:
    //    value             The value to set the flag to.
    ///////////////////////////////////////////////////////////////////////

private:
    typedef enum
    {
        STATE_FILE_HDR,
        STATE_RECORD,
        STATE_RESERVED1,
        STATE_ARCH,
        STATE_MESG_NUM_0,
        STATE_MESG_NUM_1,
        STATE_NUM_FIELDS,
        STATE_FIELD_NUM,
        STATE_FIELD_SIZE,
        STATE_FIELD_TYPE,
        STATE_NUM_DEV_FIELDS,
        STATE_DEV_FIELD_NUM,
        STATE_DEV_FIELD_SIZE,
        STATE_DEV_FIELD_INDEX,
        STATE_FIELD_DATA,
        STATE_DEV_FIELD_DATA,
        STATE_FILE_CRC_HIGH,
        STATES
    } STATE;

    typedef enum
    {
        RETURN_CONTINUE,
        RETURN_MESG,
        RETURN_MESG_DEF,
        RETURN_END_OF_FILE,
        RETURN_ERROR,
        RETURNS
    } RETURN;

    static const FIT_UINT8 DevFieldNumOffset;
    static const FIT_UINT8 DevFieldSizeOffset;
    static const FIT_UINT8 DevFieldIndexOffset;
    static const FIT_UINT16 BufferSize = 512;

    STATE state;
    FIT_BOOL hasDevData;
    FIT_UINT8 fileHdrOffset;
    FIT_UINT8 fileHdrSize;
    FIT_UINT32 fileDataSize;
    FIT_UINT32 fileBytesLeft;
    FIT_UINT16 crc;
    Mesg mesg;
    FIT_UINT8 localMesgIndex;
    MesgDefinition localMesgDefs[FIT_MAX_LOCAL_MESGS];
    FIT_UINT8 archs[FIT_MAX_LOCAL_MESGS];
    FIT_UINT8 numFields;
    FIT_UINT8 fieldIndex;
    FIT_UINT8 fieldDataIndex;
    FIT_UINT8 fieldBytesLeft;
    FIT_UINT8 fieldData[FIT_MAX_FIELD_SIZE];
    FIT_UINT8 lastTimeOffset;
    FIT_UINT32 timestamp;
    Accumulator accumulator;
    std::istream* file;
    MesgListener* mesgListener;
    MesgDefinitionListener* mesgDefinitionListener;
    DeveloperFieldDescriptionListener* descriptionListener;
    FIT_BOOL pause;
    std::string headerException;
    FIT_BOOL skipHeader;
    FIT_BOOL streamIsComplete;
    FIT_BOOL invalidDataSize;
    FIT_BOOL suppressComponentExpansion;
    FIT_UINT32 currentByteOffset;
    std::unordered_map<FIT_UINT8, DeveloperDataIdMesg> developers;
    std::unordered_map<FIT_UINT8, std::unordered_map<FIT_UINT8, FieldDescriptionMesg>> descriptions;
    FIT_UINT32 currentByteIndex;
    FIT_UINT32 bytesRead;
    char buffer[BufferSize];
    

    void InitRead(std::istream &file);
    void InitRead(std::istream &file, FIT_BOOL startOfFile);
    void UpdateEndianness(FIT_UINT8 type, FIT_UINT8 size);
    RETURN ReadByte(FIT_UINT8 data);
    void ExpandComponents(Field* containingField, const Profile::FIELD_COMPONENT* components, FIT_UINT16 numComponents);
    FIT_BOOL Read(std::istream* file);
};

} // namespace fit

#endif // defined(DECODE_HPP)


