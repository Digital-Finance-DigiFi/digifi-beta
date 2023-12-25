from typing import Union
import numpy as np
from scipy.optimize import fsolve
from enum import Enum
from src.digifi.utilities.general_utils import compare_array_len



class CompoundingType(Enum):
    CONTINUOUS = 1
    PERIODIC = 2



class Cashflow:
    """
    Base class for generating cashflow array with a base cashflow growth rate and inflation rate.
    """

    def __init__(self, cashflow: Union[np.ndarray, float], final_time: float, start_time: float=1, time_step: float=1,
                 time_array: Union[np.ndarray, None]=None, cashflow_growth_rate: float=0, inflation_rate: float=0) -> None:
        self.cashflow_growth_rate = float(cashflow_growth_rate)
        self.inflation_rate = float(inflation_rate)
        # Cashflow array is provided
        if isinstance(cashflow, np.ndarray):
            # Time array is provided
            if isinstance(time_array, np.ndarray):
                if len(time_array)!=len(cashflow):
                    raise TypeError("The length of cashflow and time_array do not match.")
                else:
                    self.time_array = time_array
            # Time array is generated
            else:
                self.time_array = self.__time_array(cashflow=cashflow, final_time=float(final_time), start_time=float(start_time),
                                                    time_step=float(time_step))
            self.cashflow = cashflow
        # Cashflow array is generated
        else:
            # Time array is provided
            if isinstance(time_array, np.ndarray):
                self.time_array = time_array
            else:
                self.time_array = self.__time_array(cashflow=float(cashflow), final_time=float(final_time), start_time=float(start_time),
                                                    time_step=float(time_step))
            self.cashflow = self.__cashflow_array(cashflow=float(cashflow))
    
    def __cashflow_array(self, cashflow: float) -> np.ndarray:
        cashflow_array = cashflow*np.ones(len(self.time_array))
        if (self.cashflow_growth_rate!=0) or (self.inflation_rate!=0):
            cashflow_array[0] = cashflow_array[0]/(1+self.inflation_rate)
            for i in range(1, len(self.time_array)):
                cashflow_array[i] = (1+self.cashflow_growth_rate)*cashflow_array[i-1]/(1+self.inflation_rate)
        return cashflow_array
    
    def __time_array(self, cashflow: Union[np.ndarray, float], final_time: float, start_time: float, time_step: float) -> np.ndarray:
        time_array = np.arange(start=start_time, stop=final_time+time_step, step=time_step)
        if isinstance(cashflow, np.ndarray) and (len(cashflow)!=len(time_array)):
            raise ValueError("The provided arguments final_time, start_time, and time_step do not generate the array of the same length as the cashflow.")
        return time_array



def present_value(cashflow: np.ndarray, time_array: np.ndarray, rate: Union[np.ndarray, float], compounding_type: CompoundingType=CompoundingType.CONTINUOUS,
                  compounding_frequency: int=1) -> float:
    """
    Present value of the cashflow discounted at a certain rate for every time period.
    """
    compare_array_len(array_1=cashflow, array_2=time_array, array_1_name="cashflow", array_2_name="time_array")
    if isinstance(rate, float) or isinstance(rate, int):
        if (rate<-1) and (1<rate):
            raise ValueError("The agrument rate must be defined in in the interval (-1,1).")
        rate = rate*np.ones(len(cashflow))
    elif isinstance(rate, np.ndarray):
        if (sum(rate<-1)!=0) or (sum(1<rate)!=0):
            raise ValueError("All rates must be in the range (-1, 1).")
    else:
        raise TypeError("The argument rate must be either of type float or np.ndarray.")
    present_value = 0
    for i in range(0, len(time_array)):
        discount_term = Compounding(rate=rate[i], compounding_type=compounding_type, compounding_frequency=compounding_frequency)
        present_value = present_value + cashflow[i]*discount_term.compounding_term(time=time_array[i])
    return present_value



def net_present_value(initial_cashflow: float, cashflow: np.ndarray, time_array: np.ndarray, rate: Union[np.ndarray, float],
                      compounding_type: CompoundingType=CompoundingType.CONTINUOUS, compounding_frequency: int=1) -> float:
    return -initial_cashflow+present_value(cashflow=cashflow, time_array=time_array, rate=rate,
                                           compounding_type=compounding_type, compounding_frequency=compounding_frequency)



def future_value(current_value: float, rate: float,  time: float, compounding_type: CompoundingType=CompoundingType.CONTINUOUS,
                 compounding_frequency: int=1) -> float:
    """
    Future value of the cashflow with a certain interest rate at a specific time.
    """
    if (rate<-1) or (1<rate):
        raise ValueError("The argument rate must be defined in in the interval [-1,1].")
    discount_term = Compounding(rate=rate, compounding_type=compounding_type, compounding_frequency=compounding_frequency)
    future_value = current_value/discount_term.compounding_term(time=time)
    return future_value



def internal_rate_of_return(initial_cashflow: float, cashflow: np.ndarray, time_array: np.ndarray,
                            compounding_type: CompoundingType=CompoundingType.CONTINUOUS, compounding_frequency: int=1) -> float:
    def cashflow_series(rate: float) -> float:
        return present_value(cashflow=cashflow, time_array=time_array, rate=float(rate), compounding_type=compounding_type,
                             compounding_frequency=compounding_frequency) - initial_cashflow
    return fsolve(cashflow_series, x0=0)[0]



def ptp_compounding_transformation(current_rate: float, current_frequency: int, new_frequency: int) -> float:
    """
    Periodic-to-periodic compounding transformation between different compounding frequencies.
    """
    new_rate = new_frequency*((1+current_rate/current_frequency)**(current_frequency/new_frequency)-1)
    return new_rate



def ptc_compounding_transformation(periodic_rate: float, periodeic_frequency: int=1) -> float:
    """
    Periodic-to-continuous compounding transformation.
    """
    continuous_rate = periodeic_frequency*np.log(1+periodic_rate/periodeic_frequency)
    return continuous_rate



def ctp_compounding_transformation(continuous_rate: float, periodic_frequency: int=1) -> float:
    """
    Continuous-to-periodic compounding transformation.
    """
    periodic_rate = periodic_frequency*(np.exp(continuous_rate/periodic_frequency)-1)
    return periodic_rate



def real_interest_rate(nominal_interest_rate: float, inflation: float) -> float:
    """
    A conversion from nominal interest rate to real interest rate based on inflation.
    """
    real_interest_rate = (1+nominal_interest_rate)/(1+inflation)-1
    return real_interest_rate



def forward_interest_rate(zero_rate_1: float, time_1: float, zero_rate_2: float, time_2: float) -> float:
    """
    Forward interest rate for the period between time_1 and time_2.
    """
    return (zero_rate_2*time_2-zero_rate_1*time_1)/(time_2-time_1)



class Compounding:
    """
    Different compounding techniques.
    Compounding term is defined as the discounting terms for future cashflows.
    """
    def __init__(self, rate: float, compounding_type: CompoundingType=CompoundingType.CONTINUOUS, compounding_frequency: int=1) -> None:
        self.rate = float(rate)
        self.compounding_type = compounding_type
        if (compounding_type==CompoundingType.PERIODIC) and (compounding_frequency<=0):
            raise ValueError("For periodic compouning the compounding_frequency must be defined.")
        self.compounding_frequency = int(compounding_frequency)
    
    def __continuous_compounding_term(self, time: float) -> float:
        return np.exp(-self.rate*time)
    
    def __periodic_compounding_term(self, time: float) -> float:
        return (1+self.rate/self.compounding_frequency)**(-self.compounding_frequency*time)
    
    def compounding_term(self, time: float) -> float:
        if self.compounding_type==CompoundingType.CONTINUOUS:
            return self.__continuous_compounding_term(time=time)
        else:
            return self.__periodic_compounding_term(time=time)

    def __ptp_compounding_transformation(self, new_frequency: int=1) -> None:
        """
        Periodic-to-continuous compounding transformation.
        """
        if self.compounding_type==CompoundingType.PERIODIC:
            self.rate = ptp_compounding_transformation(current_rate=self.rate, current_frequency=self.compounding_frequency,
                                                       new_frequency=new_frequency)
            self.compounding_frequency = new_frequency

    def __ptc_compounding_transformation(self) -> None:
        """
        Periodic-to-continuous compounding transformation.
        """
        if self.compounding_type==CompoundingType.PERIODIC:
            self.rate = ptc_compounding_transformation(periodic_rate=self.rate, periodeic_frequency=self.compounding_frequency)
            self.compounding_frequency = 1
            self.continuous = True
    
    def __ctp_compounding_transformation(self, new_frequency: int=1) -> None:
        """
        Continuous-to-periodic compounding transformation.
        """
        if self.compounding_type==CompoundingType.CONTINUOUS:
            self.rate = ctp_compounding_transformation(continuous_rate=self.rate, periodic_frequency=new_frequency)
            self.compounding_frequency = new_frequency
            self.continuous = False
    
    def compounding_transformation(self, new_compounding_type: CompoundingType=CompoundingType.PERIODIC, new_compounding_frequency: int=1) -> None:
        if (new_compounding_type==CompoundingType.PERIODIC) and (new_compounding_frequency<=0):
            raise ValueError("For periodic compouning the compounding_frequency must be defined.")
        if (new_compounding_type==CompoundingType.CONTINUOUS) and (self.compounding_type==CompoundingType.PERIODIC):
            self.rate = self.__ptc_compounding_transformation()
        if (new_compounding_type==CompoundingType.PERIODIC) and (self.compounding_type==CompoundingType.CONTINUOUS):
            self.rate = self.__ctp_compounding_transformation(new_frequency=new_compounding_frequency)
        if (new_compounding_type==CompoundingType.PERIODIC) and (self.compounding_type==CompoundingType.PERIODIC):
            self.rate = self.__ptp_compounding_transformation(new_frequency=new_compounding_frequency)
        self.compounding_type = new_compounding_type
        self.compounding_frequency = new_compounding_frequency



class Perpetuity:
    """
    A series of fixed income cashflows paid out each time step forever.
    """
    def __init__(self, perpetuity_cashflow: float, rate: float, perpetuity_growth_rate: float=0,
                 compounding_type: CompoundingType=CompoundingType.PERIODIC) -> None:
        if rate<=perpetuity_growth_rate:
            raise ValueError("The rate cannot be smaller than the perpetuity growth rate.")
        self.perpetuity_cashflow = float(perpetuity_cashflow)
        self.rate = float(rate)
        self.perpetuity_growth_rate = float(perpetuity_growth_rate)
        self.compounding_type = compounding_type

    def present_value(self) -> float:
        if self.compounding_type==CompoundingType.PERIODIC:
            pv_perpetuity = self.perpetuity_cashflow/(self.rate-self.perpetuity_growth_rate)
        else:
            pv_perpetuity = self.perpetuity_cashflow*np.exp(-self.perpetuity_growth_rate)/(np.exp(self.rate-self.perpetuity_growth_rate)-1)
        return pv_perpetuity
    
    def net_present_value(self, initial_cashflow: float) -> float:
        return -initial_cashflow + self.present_value()
    
    def future_value(self, final_time: float) -> float:
        if self.compounding_type==CompoundingType.PERIODIC:
            fv_perpetuity = self.present_value()*(1+self.rate)**(final_time)
        else:
            fv_perpetuity = self.present_value*np.exp(self.rate*final_time)
        return fv_perpetuity



class Annuity:
    """
    A series of fixed income cashflows paid out for a specified number of time periods periods.
    """
    def __init__(self, annuity_cashflow: float, rate: float, final_time: float, annuity_growth_rate: float=0,
                 compounding_type: CompoundingType=CompoundingType.PERIODIC) -> None:
        if rate<=annuity_growth_rate:
            raise ValueError("The rate cannot be smaller than the annuity growth rate.")
        self.annuity_cashflow = float(annuity_cashflow)
        self.rate = float(rate)
        self.final_time = float(final_time)
        self.annuity_growth_rate = float(annuity_growth_rate)
        self.compounding_type = compounding_type

    def present_value(self) -> float:
        if self.compounding_type==CompoundingType.PERIODIC:
            pv_annuity = self.annuity_cashflow/(self.rate-self.annuity_growth_rate)*(1-((1+self.annuity_growth_rate)/(1+self.rate))**self.final_time)
        else:
            pv_annuity = (self.annuity_cashflow*np.exp(-self.annuity_growth_rate)/(np.exp(self.rate-self.annuity_growth_rate)-1)*
                          (1-np.exp((self.annuity_growth_rate-self.rate)*self.final_time)))
        return pv_annuity
    
    def net_present_value(self, initial_cashflow: float) -> float:
        return -initial_cashflow + self.present_value()
    
    def future_value(self) -> float:
        if self.compounding_type==CompoundingType.PERIODIC:
            fv_annuity = self.present_value()*(1+self.rate)**(self.final_time)
        else:       
            fv_annuity = self.present_value()*np.exp(self.rate*self.final_time)
        return fv_annuity