import { createContext, useContext, useState, ReactNode } from 'react';

export interface Filters {
  ddos: boolean;
  ransomware_with_data_theft: boolean;
  data_exfiltration: boolean;
  wiper: boolean;
  fraud: boolean;
  defacement: boolean;
  cve: boolean;
}

interface FilterContextType {
  filters: Filters;
  setFilters: (filters: Filters | ((prev: Filters) => Filters)) => void;
}

const FilterContext = createContext<FilterContextType | undefined>(undefined);

export const useFilterContext = () => {
  const context = useContext(FilterContext);
  if (context === undefined) {
    throw new Error('useFilterContext must be used within a FilterProvider');
  }
  return context;
};

interface FilterProviderProps {
  children: ReactNode;
}

export const FilterProvider = ({ children }: FilterProviderProps) => {
  const [filters, setFilters] = useState<Filters>({
    ddos: true,
    ransomware_with_data_theft: true,
    data_exfiltration: true,
    wiper: true,
    fraud: true,
    defacement: true,
    cve: true,
  });

  return (
    <FilterContext.Provider value={{ filters, setFilters }}>
      {children}
    </FilterContext.Provider>
  );
}; 