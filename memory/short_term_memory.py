```json
{
    "memory/short_term_memory.py": {
        "content": "
import logging
from typing import List, Dict
from attention_is_all_you_need_pytorch import StateGraph
from AI_Researcher import Letta

class ShortTermMemory:
    """
    A class representing short-term memory in the Electrochemical Risk Navigator.
    
    Attributes:
    non_stationary_drift_index (float): The index of non-stationary drift in the memory.
    stochastic_regime_switch (bool): Whether the memory is in a stochastic regime switch.
    """

    def __init__(self, non_stationary_drift_index: float, stochastic_regime_switch: bool):
        """
        Initializes the ShortTermMemory object.
        
        Args:
        non_stationary_drift_index (float): The index of non-stationary drift in the memory.
        stochastic_regime_switch (bool): Whether the memory is in a stochastic regime switch.
        """
        self.non_stationary_drift_index = non_stationary_drift_index
        self.stochastic_regime_switch = stochastic_regime_switch
        self.logger = logging.getLogger(__name__)

    def update_memory(self, new_data: List[Dict]) -> None:
        """
        Updates the short-term memory with new data.
        
        Args:
        new_data (List[Dict]): The new data to update the memory with.
        
        Raises:
        Exception: If an error occurs during the update process.
        """
        try:
            self.logger.info('Updating short-term memory...')
            # Use LangGraph's StateGraph to manage memory
            state_graph = StateGraph()
            state_graph.update_state(new_data)
            self.logger.info('Short-term memory updated successfully.')
        except Exception as e:
            self.logger.error(f'Error updating short-term memory: {e}')
            raise

    def manage_memory(self) -> None:
        """
        Manages the short-term memory using Letta's memory management.
        
        Raises:
        Exception: If an error occurs during the management process.
        """
        try:
            self.logger.info('Managing short-term memory...')
            # Use Letta's memory management
            letta = Letta()
            letta.manage_memory(self.non_stationary_drift_index, self.stochastic_regime_switch)
            self.logger.info('Short-term memory managed successfully.')
        except Exception as e:
            self.logger.error(f'Error managing short-term memory: {e}')
            raise

if __name__ == '__main__':
    # Simulate the 'Rocket Science' problem
    non_stationary_drift_index = 0.5
    stochastic_regime_switch = True
    short_term_memory = ShortTermMemory(non_stationary_drift_index, stochastic_regime_switch)
    new_data = [{'key': 'value'}]
    short_term_memory.update_memory(new_data)
    short_term_memory.manage_memory()
",
        "commit_message": "feat: implement specialized short_term_memory logic"
    }
}
```