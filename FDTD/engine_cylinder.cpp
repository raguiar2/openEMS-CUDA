/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "engine_cylinder.h"

Engine_Cylinder* Engine_Cylinder::New(const Operator_Cylinder* op)
{
	Engine_Cylinder* e = new Engine_Cylinder(op);
	e->Init();
	return e;
}

Engine_Cylinder::Engine_Cylinder(const Operator_Cylinder* op) : Engine(op)
{
	cyl_Op = op;
	if (cyl_Op->GetClosedAlpha())
	{
		++numLines[1]; //necessary for dobled voltage and current line in alpha-dir, operator will return one smaller for correct post-processing
	}
}

Engine_Cylinder::~Engine_Cylinder()
{
	Reset();
}

void Engine_Cylinder::Init()
{
	Engine::Init();

//	if (cyl_Op->GetClosedAlpha())
//	{
//		unsigned int lastLine = Op->numLines[1]-1; //number of last alpha-line
//		unsigned int pos[3];
//		for (pos[0]=0;pos[0]<Op->numLines[0];++pos[0])
//		{
//			for (int n=0;n<3;++n)
//			{
//				delete[] volt[n][pos[0]][lastLine];
//				volt[n][pos[0]][lastLine] = volt[n][pos[0]][0];
//				delete[] curr[n][pos[0]][lastLine];
//				curr[n][pos[0]][lastLine] = curr[n][pos[0]][0];
//			}
//		}
//	}
}

void Engine_Cylinder::Reset()
{
//	if (cyl_Op->GetClosedAlpha())
//	{
//		unsigned int lastLine = Op->numLines[1]-1; //number of last alpha-line
//		unsigned int pos[3];
//		for (pos[0]=0;pos[0]<Op->numLines[0];++pos[0])
//		{
//			for (int n=0;n<3;++n)
//			{
//				volt[n][pos[0]][lastLine] = NULL;
//				curr[n][pos[0]][lastLine] = NULL;
//			}
//		}
//	}
}

inline void Engine_Cylinder::CloseAlphaVoltages()
{
	unsigned int pos[3];
	// copy voltages from last alpha-plane to first
	unsigned int last_A_Line = numLines[1]-1;
	for (pos[0]=0;pos[0]<numLines[0];++pos[0])
	{
		for (pos[2]=0;pos[2]<numLines[2];++pos[2])
		{
			volt[0][pos[0]][0][pos[2]] = volt[0][pos[0]][last_A_Line][pos[2]];
			volt[1][pos[0]][0][pos[2]] = volt[1][pos[0]][last_A_Line][pos[2]];
			volt[2][pos[0]][0][pos[2]] = volt[2][pos[0]][last_A_Line][pos[2]];
		}

	}
}

inline void Engine_Cylinder::CloseAlphaCurrents()
{
	unsigned int pos[3];
	// copy currents from first alpha-plane to last
	for (pos[0]=0;pos[0]<numLines[0]-1;++pos[0])
	{
		unsigned int last_A_Line = numLines[1]-1;
		for (pos[2]=0;pos[2]<numLines[2]-1;++pos[2])
		{
			curr[0][pos[0]][last_A_Line][pos[2]] = curr[0][pos[0]][0][pos[2]];
			curr[1][pos[0]][last_A_Line][pos[2]] = curr[1][pos[0]][0][pos[2]];
			curr[2][pos[0]][last_A_Line][pos[2]] = curr[2][pos[0]][0][pos[2]];
		}
	}
}

bool Engine_Cylinder::IterateTS(unsigned int iterTS)
{
	if (cyl_Op->GetClosedAlpha()==false)
		return Engine::IterateTS(iterTS);

	for (unsigned int iter=0;iter<iterTS;++iter)
	{
		UpdateVoltages();
		ApplyVoltageExcite();

		CloseAlphaVoltages();

		UpdateCurrents();
		ApplyCurrentExcite();

		CloseAlphaCurrents();

		++numTS;
	}

	return true;
}


//inline void Engine_Cylinder::UpdateVoltages()
//{
//	unsigned int pos[3];
//	bool shift[3];
//
//	if (cyl_Op->GetClosedAlpha()==false)
//		return Engine::UpdateVoltages();
//
//	//voltage updates
//	for (pos[0]=0;pos[0]<Op->numLines[0];++pos[0])
//	{
//		shift[0]=pos[0];
//		for (pos[1]=1;pos[1]<Op->numLines[1];++pos[1])
//		{
//			shift[1]=pos[1];
//			for (pos[2]=0;pos[2]<Op->numLines[2];++pos[2])
//			{
//				shift[2]=pos[2];
//				//do the updates here
//				//for x
//				volt[0][pos[0]][pos[1]][pos[2]] *= Op->vv[0][pos[0]][pos[1]][pos[2]];
//				volt[0][pos[0]][pos[1]][pos[2]] += Op->vi[0][pos[0]][pos[1]][pos[2]] * ( curr[2][pos[0]][pos[1]][pos[2]] - curr[2][pos[0]][pos[1]-shift[1]][pos[2]] - curr[1][pos[0]][pos[1]][pos[2]] + curr[1][pos[0]][pos[1]][pos[2]-shift[2]]);
//
//				//for y
//				volt[1][pos[0]][pos[1]][pos[2]] *= Op->vv[1][pos[0]][pos[1]][pos[2]];
//				volt[1][pos[0]][pos[1]][pos[2]] += Op->vi[1][pos[0]][pos[1]][pos[2]] * ( curr[0][pos[0]][pos[1]][pos[2]] - curr[0][pos[0]][pos[1]][pos[2]-shift[2]] - curr[2][pos[0]][pos[1]][pos[2]] + curr[2][pos[0]-shift[0]][pos[1]][pos[2]]);
//
//				//for z
//				volt[2][pos[0]][pos[1]][pos[2]] *= Op->vv[2][pos[0]][pos[1]][pos[2]];
//				volt[2][pos[0]][pos[1]][pos[2]] += Op->vi[2][pos[0]][pos[1]][pos[2]] * ( curr[1][pos[0]][pos[1]][pos[2]] - curr[1][pos[0]-shift[0]][pos[1]][pos[2]] - curr[0][pos[0]][pos[1]][pos[2]] + curr[0][pos[0]][pos[1]-shift[1]][pos[2]]);
//			}
//		}
//
//		// copy voltages from last alpha-plane to first
//		unsigned int last_A_Line = Op->numLines[1]-1;
//		for (pos[2]=0;pos[2]<Op->numLines[2];++pos[2])
//		{
//			volt[0][pos[0]][0][pos[2]] = volt[0][pos[0]][last_A_Line][pos[2]];
//			volt[1][pos[0]][0][pos[2]] = volt[1][pos[0]][last_A_Line][pos[2]];
//			volt[2][pos[0]][0][pos[2]] = volt[2][pos[0]][last_A_Line][pos[2]];
//		}
//
//	}
//}
//
//inline void Engine_Cylinder::UpdateCurrents()
//{
//	if (cyl_Op->GetClosedAlpha()==false)
//		return Engine::UpdateCurrents();
//
//	unsigned int pos[3];
//	for (pos[0]=0;pos[0]<Op->numLines[0]-1;++pos[0])
//	{
//		for (pos[1]=0;pos[1]<Op->numLines[1]-1;++pos[1])
//		{
//			for (pos[2]=0;pos[2]<Op->numLines[2]-1;++pos[2])
//			{
//				//do the updates here
//				//for x
//				curr[0][pos[0]][pos[1]][pos[2]] *= Op->ii[0][pos[0]][pos[1]][pos[2]];
//				curr[0][pos[0]][pos[1]][pos[2]] += Op->iv[0][pos[0]][pos[1]][pos[2]] * ( volt[2][pos[0]][pos[1]][pos[2]] - volt[2][pos[0]][pos[1]+1][pos[2]] - volt[1][pos[0]][pos[1]][pos[2]] + volt[1][pos[0]][pos[1]][pos[2]+1]);
//
//				//for y
//				curr[1][pos[0]][pos[1]][pos[2]] *= Op->ii[1][pos[0]][pos[1]][pos[2]];
//				curr[1][pos[0]][pos[1]][pos[2]] += Op->iv[1][pos[0]][pos[1]][pos[2]] * ( volt[0][pos[0]][pos[1]][pos[2]] - volt[0][pos[0]][pos[1]][pos[2]+1] - volt[2][pos[0]][pos[1]][pos[2]] + volt[2][pos[0]+1][pos[1]][pos[2]]);
//
//				//for z
//				curr[2][pos[0]][pos[1]][pos[2]] *= Op->ii[2][pos[0]][pos[1]][pos[2]];
//				curr[2][pos[0]][pos[1]][pos[2]] += Op->iv[2][pos[0]][pos[1]][pos[2]] * ( volt[1][pos[0]][pos[1]][pos[2]] - volt[1][pos[0]+1][pos[1]][pos[2]] - volt[0][pos[0]][pos[1]][pos[2]] + volt[0][pos[0]][pos[1]+1][pos[2]]);
//			}
//		}
//		// copy currents from first alpha-plane to last
//		unsigned int last_A_Line = Op->numLines[1]-1;
//		for (pos[2]=0;pos[2]<Op->numLines[2]-1;++pos[2])
//		{
//			curr[0][pos[0]][last_A_Line][pos[2]] = curr[0][pos[0]][0][pos[2]];
//			curr[1][pos[0]][last_A_Line][pos[2]] = curr[1][pos[0]][0][pos[2]];
//			curr[2][pos[0]][last_A_Line][pos[2]] = curr[2][pos[0]][0][pos[2]];
//		}
//	}
//}