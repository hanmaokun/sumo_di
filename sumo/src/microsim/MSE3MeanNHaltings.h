#ifndef MSE3MEANNHALTINGS_H
#define MSE3MEANNHALTINGS_H

///
/// @file    MSE3MeanNHaltings.h
/// @author  Christian Roessel <christian.roessel@dlr.de>
/// @date    Started Fri Nov 28 2003 12:27 CET
/// @version $Id$
///
/// @brief   
///
///

/* Copyright (C) 2003 by German Aerospace Center (http://www.dlr.de) */

//---------------------------------------------------------------------------//
//
//   This program is free software; you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation; either version 2 of the License, or
//   (at your option) any later version.
//
//---------------------------------------------------------------------------//

#include "MSDetectorHaltingContainerWrapper.h"

class MSVehicle;

class MSE3MeanNHaltings
{
public:

protected:
    typedef double DetectorAggregate;
    typedef DetectorContainer::HaltingsMap Container;
    typedef Container::InnerContainer HaltingsMap;
    
    MSE3MeanNHaltings( const Container& container )
        :
        containerM( container )
        {}

    virtual ~MSE3MeanNHaltings( void ) 
        {}

    bool hasVehicle( MSVehicle& veh ) const
        {
            return containerM.hasVehicle( &veh );
        }

    DetectorAggregate getAggregate( MSVehicle& veh ) // [nHalts]
        {
            HaltingsMap haltMap = containerM.containerM;
            HaltingsMap::const_iterator pair = haltMap.find( &veh );
            assert( pair != haltMap.end() );
            return pair->second.nHalts;
        }

    static std::string getDetectorName( void )
        {
            return "meanNHaltsPerVehicle";
        }

private:
    const Container& containerM;

    MSE3MeanNHaltings();
    MSE3MeanNHaltings( const MSE3MeanNHaltings& );
    MSE3MeanNHaltings& operator=( const MSE3MeanNHaltings& );
};


// Local Variables:
// mode:C++
// End:

#endif // MSE3MEANNHALTINGS_H
