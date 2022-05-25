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
    /// Implements the StressLevel profile message.
    /// </summary>
    public class StressLevelMesg : Mesg
    {
        #region Fields
        #endregion

        /// <summary>
        /// Field Numbers for <see cref="StressLevelMesg"/>
        /// </summary>
        public sealed class FieldDefNum
        {
            public const byte StressLevelValue = 0;
            public const byte StressLevelTime = 1;
            public const byte Invalid = Fit.FieldNumInvalid;
        }

        #region Constructors
        public StressLevelMesg() : base(Profile.GetMesg(MesgNum.StressLevel))
        {
        }

        public StressLevelMesg(Mesg mesg) : base(mesg)
        {
        }
        #endregion // Constructors

        #region Methods
        ///<summary>
        /// Retrieves the StressLevelValue field</summary>
        /// <returns>Returns nullable short representing the StressLevelValue field</returns>
        public short? GetStressLevelValue()
        {
            Object val = GetFieldValue(0, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return (Convert.ToInt16(val));
            
        }

        /// <summary>
        /// Set StressLevelValue field</summary>
        /// <param name="stressLevelValue_">Nullable field value to be set</param>
        public void SetStressLevelValue(short? stressLevelValue_)
        {
            SetFieldValue(0, 0, stressLevelValue_, Fit.SubfieldIndexMainField);
        }
        
        ///<summary>
        /// Retrieves the StressLevelTime field
        /// Units: s
        /// Comment: Time stress score was calculated</summary>
        /// <returns>Returns DateTime representing the StressLevelTime field</returns>
        public DateTime GetStressLevelTime()
        {
            Object val = GetFieldValue(1, 0, Fit.SubfieldIndexMainField);
            if(val == null)
            {
                return null;
            }

            return TimestampToDateTime(Convert.ToUInt32(val));
            
        }

        /// <summary>
        /// Set StressLevelTime field
        /// Units: s
        /// Comment: Time stress score was calculated</summary>
        /// <param name="stressLevelTime_">Nullable field value to be set</param>
        public void SetStressLevelTime(DateTime stressLevelTime_)
        {
            SetFieldValue(1, 0, stressLevelTime_.GetTimeStamp(), Fit.SubfieldIndexMainField);
        }
        
        #endregion // Methods
    } // Class
} // namespace
