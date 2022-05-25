#region Copyright
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

#endregion

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.IO;
using System.Linq;

namespace Dynastream.Fit
{
    /// <summary>
    /// Implements the AntTx profile message.
    /// </summary>
    public class AntTxMesg : Mesg
    {
        #region Fields
        #endregion

        /// <summary>
        /// Field Numbers for <see cref="AntTxMesg"/>
        /// </summary>
        public sealed class FieldDefNum
        {
            public const byte Timestamp = 253;
            public const byte FractionalTimestamp = 0;
            public const byte MesgId = 1;
            public const byte MesgData = 2;
            public const byte ChannelNumber = 3;
            public const byte Data = 4;
            public const byte Invalid = Fit.FieldNumInvalid;
        }

        #region Constructors
        public AntTxMesg() : base(Profile.GetMesg(MesgNum.AntTx))
        {
        }

        public AntTxMesg(Mesg mesg) : base(mesg)
        {
        }
        #endregion // Constructors

        #region Methods
        ///<summary>
        /// Retrieves the Timestamp field
        /// Units: s</summary>
        /// <returns>Returns DateTime representing the Timestamp field</returns>
        public DateTime GetTimestamp()
        {
            Object val = GetFieldValue(253, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return TimestampToDateTime(Convert.ToUInt32(val));
            
        }

        /// <summary>
        /// Set Timestamp field
        /// Units: s</summary>
        /// <param name="timestamp_">Nullable field value to be set</param>
        public void SetTimestamp(DateTime timestamp_)
        {
            SetFieldValue(253, 0, timestamp_.GetTimeStamp(), Fit.SubfieldIndexMainField);
        }
        
        ///<summary>
        /// Retrieves the FractionalTimestamp field
        /// Units: s</summary>
        /// <returns>Returns nullable float representing the FractionalTimestamp field</returns>
        public float? GetFractionalTimestamp()
        {
            Object val = GetFieldValue(0, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToSingle(val));
            
        }

        /// <summary>
        /// Set FractionalTimestamp field
        /// Units: s</summary>
        /// <param name="fractionalTimestamp_">Nullable field value to be set</param>
        public void SetFractionalTimestamp(float? fractionalTimestamp_)
        {
            SetFieldValue(0, 0, fractionalTimestamp_, Fit.SubfieldIndexMainField);
        }
        
        ///<summary>
        /// Retrieves the MesgId field</summary>
        /// <returns>Returns nullable byte representing the MesgId field</returns>
        public byte? GetMesgId()
        {
            Object val = GetFieldValue(1, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToByte(val));
            
        }

        /// <summary>
        /// Set MesgId field</summary>
        /// <param name="mesgId_">Nullable field value to be set</param>
        public void SetMesgId(byte? mesgId_)
        {
            SetFieldValue(1, 0, mesgId_, Fit.SubfieldIndexMainField);
        }
        
        
        /// <summary>
        ///
        /// </summary>
        /// <returns>returns number of elements in field MesgData</returns>
        public int GetNumMesgData()
        {
            return GetNumFieldValues(2, Fit.SubfieldIndexMainField);
        }

        ///<summary>
        /// Retrieves the MesgData field</summary>
        /// <param name="index">0 based index of MesgData element to retrieve</param>
        /// <returns>Returns nullable byte representing the MesgData field</returns>
        public byte? GetMesgData(int index)
        {
            Object val = GetFieldValue(2, index, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToByte(val));
            
        }

        /// <summary>
        /// Set MesgData field</summary>
        /// <param name="index">0 based index of mesg_data</param>
        /// <param name="mesgData_">Nullable field value to be set</param>
        public void SetMesgData(int index, byte? mesgData_)
        {
            SetFieldValue(2, index, mesgData_, Fit.SubfieldIndexMainField);
        }
        
        ///<summary>
        /// Retrieves the ChannelNumber field</summary>
        /// <returns>Returns nullable byte representing the ChannelNumber field</returns>
        public byte? GetChannelNumber()
        {
            Object val = GetFieldValue(3, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToByte(val));
            
        }

        /// <summary>
        /// Set ChannelNumber field</summary>
        /// <param name="channelNumber_">Nullable field value to be set</param>
        public void SetChannelNumber(byte? channelNumber_)
        {
            SetFieldValue(3, 0, channelNumber_, Fit.SubfieldIndexMainField);
        }
        
        
        /// <summary>
        ///
        /// </summary>
        /// <returns>returns number of elements in field Data</returns>
        public int GetNumData()
        {
            return GetNumFieldValues(4, Fit.SubfieldIndexMainField);
        }

        ///<summary>
        /// Retrieves the Data field</summary>
        /// <param name="index">0 based index of Data element to retrieve</param>
        /// <returns>Returns nullable byte representing the Data field</returns>
        public byte? GetData(int index)
        {
            Object val = GetFieldValue(4, index, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToByte(val));
            
        }

        /// <summary>
        /// Set Data field</summary>
        /// <param name="index">0 based index of data</param>
        /// <param name="data_">Nullable field value to be set</param>
        public void SetData(int index, byte? data_)
        {
            SetFieldValue(4, index, data_, Fit.SubfieldIndexMainField);
        }
        
        #endregion // Methods
    } // Class
} // namespace
