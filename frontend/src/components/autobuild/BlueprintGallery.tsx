import BlueprintCard from "@/components/autobuild/BlueprintCard";
import { Blueprint } from "@/types/blueprint";

interface BlueprintGalleryProps {
  blueprints: Blueprint[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export default function BlueprintGallery({ blueprints, selectedId, onSelect }: BlueprintGalleryProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {blueprints.map((blueprint) => (
        <BlueprintCard
          key={blueprint.id}
          blueprint={blueprint}
          selected={blueprint.id === selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
