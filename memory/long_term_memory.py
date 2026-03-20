```json
{
    "memory/long_term_memory.py": {
        "content": "
import logging
from typing import Dict, List
from attention_is_all_you_need_pytorch import StateGraph
from langroid import LangGraph

class LongTermMemory:
    """
    A class to manage long-term memory for the Electrochemical Risk Navigator.
    
    Attributes:
    non_stationary_drift_index (Dict): A dictionary to track non-stationary drift in the data.
    stochastic_regime_switch (List): A list to store stochastic regime switch data.
    """

    def __init__(self, non_stationary_drift_index: Dict, stochastic_regime_switch: List):
        """
        Initialize the LongTermMemory class.
        
        Args:
        non_stationary_drift_index (Dict): A dictionary to track non-stationary drift in the data.
        stochastic_regime_switch (List): A list to store stochastic regime switch data.
        """
        self.non_stationary_drift_index = non_stationary_drift_index
        self.stochastic_regime_switch = stochastic_regime_switch
        self.logger = logging.getLogger(__name__)

    def update_non_stationary_drift_index(self, new_data: Dict) -> None:
        """
        Update the non-stationary drift index with new data.
        
        Args:
        new_data (Dict): New data to update the non-stationary drift index.
        
        Raises:
        Exception: If an error occurs while updating the non-stationary drift index.
        """
        try:
            self.non_stationary_drift_index.update(new_data)
            self.logger.info('Non-stationary drift index updated successfully')
        except Exception as e:
            self.logger.error(f'Error updating non-stationary drift index: {e}')

    def update_stochastic_regime_switch(self, new_data: List) -> None:
        """
        Update the stochastic regime switch data with new data.
        
        Args:
        new_data (List): New data to update the stochastic regime switch.
        
        Raises:
        Exception: If an error occurs while updating the stochastic regime switch data.
        """
        try:
            self.stochastic_regime_switch.extend(new_data)
            self.logger.info('Stochastic regime switch data updated successfully')
        except Exception as e:
            self.logger.error(f'Error updating stochastic regime switch data: {e}')

    def get_state_graph(self) -> StateGraph:
        """
        Get the state graph for the long-term memory.
        
        Returns:
        StateGraph: The state graph for the long-term memory.
        """
        try:
            state_graph = StateGraph(self.non_stationary_drift_index, self.stochastic_regime_switch)
            self.logger.info('State graph retrieved successfully')
            return state_graph
        except Exception as e:
            self.logger.error(f'Error retrieving state graph: {e}')

    def get_lang_graph(self) -> LangGraph:
        """
        Get the language graph for the long-term memory.
        
        Returns:
        LangGraph: The language graph for the long-term memory.
        """
        try:
            lang_graph = LangGraph(self.non_stationary_drift_index, self.stochastic_regime_switch)
            self.logger.info('Language graph retrieved successfully')
            return lang_graph
        except Exception as e:
            self.logger.error(f'Error retrieving language graph: {e}')

if __name__ == '__main__':
    # Simulation of the 'Rocket Science' problem
    non_stationary_drift_index = {'drift': 0.5}
    stochastic_regime_switch = [0.2, 0.3, 0.4]
    long_term_memory = LongTermMemory(non_stationary_drift_index, stochastic_regime_switch)
    long_term_memory.update_non_stationary_drift_index({'drift': 0.6})
    long_term_memory.update_stochastic_regime_switch([0.5, 0.6])
    state_graph = long_term_memory.get_state_graph()
    lang_graph = long_term_memory.get_lang_graph()
    print(state_graph)
    print(lang_graph)
",
        "commit_message": "feat: implement specialized long_term_memory logic"
    }
}
```