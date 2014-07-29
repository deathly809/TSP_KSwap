import java.io.File;
import java.util.Scanner;
import java.util.Arrays;
import java.util.Comparator;

public class Main {

	public static void sort(File files[]) {
		Comparator<File> comp = new Comparator<File>() {
			public int compare(File a, File b) {
			try{
				String name_a[] = a.getName().split("_");
				String name_b[] = b.getName().split("_");
				
				int a_threads = Integer.parseInt(name_a[1]);
				int b_threads = Integer.parseInt(name_b[1]);
				
				if(a_threads != b_threads) return a_threads - b_threads;
				
				int a_tilesize = Integer.parseInt(name_a[2].split("\\.")[0]);
				int b_tilesize = Integer.parseInt(name_b[2].split("\\.")[0]);
				
				return a_tilesize - b_tilesize;
				
			}catch(Exception ex) {
			}
			return a.getName().compareTo(b.getName());
		}
		};
		Arrays.<File>sort(files,comp);
	}

	public static void main(String[] args) throws Exception{
		final File folder = new File(".");
		File files[] = folder.listFiles();
		sort(files);
		for(final File f : files ) {
			if(!f.getName().contains("output")) {
				continue;
			}
			
			//System.out.println(f.getName());
			
			
			Scanner input = new Scanner(f);
			int linesRead = 0;
			for(linesRead = 0; linesRead < 6; ++linesRead) {
				if(input.hasNextLine() ) {
					input.nextLine();
				}else {
					break;
				}
			}
			if(linesRead == 6) {
				String spl = input.nextLine().split(":")[1];
				System.out.println(spl.substring(1,spl.length()-1));
			}
			
		}
	}

}